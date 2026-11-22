# ATS Resume Extractor: Production Promotion Gate Audit (Stage 4 -> Stage 5)
**Audit Date:** 2026-11-18T08:06:27.221900+00:00 | **Overall Recommendation:** **BLOCKED - GATES FAILED**

## Summary of Promotion Gates
| Gate # | Promotion Requirement | Current Metric | Gate Status |
| :--- | :--- | :--- | :--- |
| **Gate 1** | Labeled Set: SLM >= Legacy across all critical fields & bug classes | SLM is >= Legacy across all critical fields. | **🟢 PASS** |
| **Gate 2** | Reliability: Invalid JSON + Upstream Timeouts < 1.0% | Error Rate: 0.0% | **🟢 PASS** |
| **Gate 3** | Memory Safety: Zero Memory Guard Incidents on Space | Trips: 0 | **🟢 PASS** |
| **Gate 4** | Grounding: Hallucination Rate < 2.0% | Hallucination Rate: 100.0% | **🔴 FAIL** |
| **Gate 5** | Downstream Integrity: Ranking Impact Verified | Spearman Rank Corr: 1.0 | **🟢 PASS** |

## Gate 1 Breakdown: Field-Level Agreement & Known Bug Classes
| Field | SLM Accuracy | Legacy Accuracy | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **candidateName** | 100.0% | 100.0% | +0.0% | 🟢 PASS |
| **email** | 100.0% | 75.0% | +25.0% | 🟢 PASS |
| **phone** | 100.0% | 100.0% | +0.0% | 🟢 PASS |
| **jobTitle** | 100.0% | 66.67% | +33.33% | 🟢 PASS |
| **totalExperienceYears** | 100.0% | 100.0% | +0.0% | 🟢 PASS |

### Known Bug Classes Status:
- **Title Truncation (Parentheses/Stacks):** 4 fixed / preserved by SLM (PASS)
- **Em-dash '—' in Emails:** 3 invalid email placeholders eliminated (PASS)
- **Null / String None Handling:** Fully normalized without string literal 'None' (PASS)

## Next Steps for Stage 5 Cutover
1. Configure canary traffic roll-out via `EXTRACTION_CUTOVER_PERCENTAGE` (10% -> 50% -> 100%).
2. Ensure instant rollback mechanism (`EXTRACTION_CUTOVER_PERCENTAGE=0`) is documented in operational runbook.
3. Keep permanent deterministic fallback to legacy rules active on any SLM exception.