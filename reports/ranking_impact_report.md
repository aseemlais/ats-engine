# ATS Resume Extractor: Downstream Ranking Impact Audit Report
**Audit Timestamp:** 2026-11-18T05:36:02.253902+00:00 | **Evaluated Candidates:** 4
**Target Role:** Senior Distributed Systems & Backend Engineer

## 1. High-Level Downstream Alignment Metrics
| Alignment Metric | Observed Value | Production Guardrail Target | Status |
| :--- | :--- | :--- | :--- |
| **Spearman Rank Correlation (rho)** | **1.0** | >= 0.850 | 🟢 PASS |
| **Score Pearson Correlation (r)** | **0.9996** | >= 0.900 | 🟢 PASS |
| **Shortlist Bracket Agreement** | **100.0%** | >= 85.0% | 🟢 PASS |
| **Mean Absolute Score Delta** | **0.83 pts** | <= 5.0 pts | 🟢 PASS |
| **Max Absolute Score Delta** | **2.94 pts** | <= 20.0 pts | 🟢 PASS |

## 2. Large Score Shifts & Rank Dislocations (|Score Delta| >= 10.0 or |Rank Shift| >= 3)
🟢 **No large score shifts detected.** Candidate rank order and scores remain stable and consistent.

## 3. Full Candidate Comparison Matrix
| Candidate ID | Name | Legacy Rank (Score) | SLM Rank (Score) | Score Delta | Status (Leg -> SLM) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CAND_001 | Priya Nair | #1 (77.55) | #1 (80.49) | +2.94 | Fit -> Fit |
| CAND_002 | David Chen | #2 (57.99) | #2 (57.62) | -0.37 | Review -> Review |
| CAND_003 | CAND_003 | #4 (0.0) | #4 (0.0) | 0.0 | Filtered Out -> Filtered Out |
| CAND_004 | CAND_004 | #4 (0.0) | #4 (0.0) | 0.0 | Filtered Out -> Filtered Out |