"""
tests/test_evaluation_tooling_phase4.py
=======================================
Acceptance & Unit Test Suite for Phase 4: Evaluation Tooling & Promotion Gates.

Validates:
  1. Shadow metrics computation (agreements, slice breakdown, errors, grounding, latency percentiles)
  2. Stratified review sampling (prioritizing errors, hallucinations, disagreements + control slice)
  3. Disagreement labeling interface (saving labels, updating overall judgments)
  4. 5 Promotion gates evaluation (labeled accuracy, bug class fixes, error bounds, memory, grounding)
  5. Downstream ranking impact evaluation (Spearman rho, bracket stability, zero scoring regressions)
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from app.models.job import JobDescription
from scripts.evaluate_shadow_metrics import (
    analyze_shadow_logs,
    generate_markdown_report,
    percentile,
)
from scripts.generate_review_sample import (
    classify_log_for_review,
    generate_stratified_sample,
)
from scripts.label_disagreements import (
    apply_batch_labels,
    load_review_labels,
    save_review_labels,
)
from scripts.check_promotion_gates import (
    evaluate_gate_1_labeled_accuracy,
    evaluate_gate_2_error_rate,
    evaluate_gate_3_memory_safety,
    evaluate_gate_4_grounding,
    evaluate_gate_5_ranking_impact,
    run_promotion_gate_audit,
)
from scripts.evaluate_ranking_impact import (
    compute_pearson_correlation,
    compute_spearman_rank_correlation,
    load_sample_resumes_corpus,
    run_ranking_impact_audit,
)


# ---------------------------------------------------------------------------
# 1. Shadow Metrics Analysis Tests
# ---------------------------------------------------------------------------

def test_percentile_calculation():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(vals, 50.0) == 30.0
    assert percentile(vals, 95.0) == 48.0
    assert percentile([], 50.0) == 0.0


def test_analyze_shadow_logs_metrics():
    logs = [
        {
            "fileHash": "hash1",
            "docFormat": "pdf",
            "pageCount": 1,
            "isTwoColumn": False,
            "charLength": 1500,
            "fieldAgreements": {
                "emailExactMatch": True,
                "phoneExactMatch": True,
                "nameTokenF1": 1.0,
                "titleTokenF1": 0.90,
                "skillsJaccard": 0.80,
                "skillsPrecision": 0.85,
                "skillsRecall": 0.90,
                "expDeltaYears": 0.0,
                "educationCountMatch": True,
                "workExperienceCountMatch": True,
            },
            "validityFlags": {
                "emailValid": True,
                "phoneValid": True,
                "namePlausible": True,
                "datesParse": True,
            },
            "groundingFlags": {
                "allGrounded": True,
                "ungroundedEntities": [],
            },
            "slmLatencyMs": 1200.0,
            "isValidJson": True,
            "errorFlags": [],
        },
        {
            "fileHash": "hash2",
            "docFormat": "docx",
            "pageCount": 2,
            "isTwoColumn": True,
            "charLength": 3200,
            "fieldAgreements": {
                "emailExactMatch": False,
                "phoneExactMatch": True,
                "nameTokenF1": 0.80,
                "titleTokenF1": 0.60,
                "skillsJaccard": 0.50,
                "skillsPrecision": 0.60,
                "skillsRecall": 0.70,
                "expDeltaYears": 1.5,
                "educationCountMatch": False,
                "workExperienceCountMatch": True,
            },
            "validityFlags": {
                "emailValid": False,
                "phoneValid": True,
                "namePlausible": True,
                "datesParse": True,
            },
            "groundingFlags": {
                "allGrounded": False,
                "ungroundedEntities": ["skill:QuantumComputing"],
            },
            "slmLatencyMs": 1800.0,
            "isValidJson": True,
            "errorFlags": ["finish_reason_length"],
        },
    ]

    metrics = analyze_shadow_logs(logs)
    assert metrics["totalRuns"] == 2
    assert metrics["agreements"]["emailExactMatchPct"] == 50.0
    assert metrics["agreements"]["phoneExactMatchPct"] == 100.0
    assert metrics["agreements"]["avgNameTokenF1"] == 0.90
    assert metrics["agreements"]["avgTitleTokenF1"] == 0.75
    assert metrics["agreements"]["avgSkillsJaccard"] == 0.65
    assert metrics["agreements"]["avgExpDeltaYears"] == 0.75

    assert metrics["errors"]["invalidJsonCount"] == 0
    assert metrics["grounding"]["groundingFailureCount"] == 1
    assert metrics["grounding"]["groundingFailureRatePct"] == 50.0
    assert metrics["latency"]["p50Ms"] == 1500.0

    # Test report formatting
    report = generate_markdown_report(metrics)
    assert "Phase 4 Shadow Mode Evaluation Report" in report
    assert "Total Shadow Evaluations:** 2" in report


# ---------------------------------------------------------------------------
# 2. Stratified Sampling Tests
# ---------------------------------------------------------------------------

def test_classify_log_for_review():
    # Error classification
    err_log = {"errorFlags": ["json_invalid"], "isValidJson": False}
    cat, reasons = classify_log_for_review(err_log)
    assert cat == "error"
    assert "invalid_json_format" in reasons

    # Hallucination classification
    halluc_log = {"groundingFlags": {"allGrounded": False, "ungroundedEntities": ["company:FakeCorp"]}}
    cat, reasons = classify_log_for_review(halluc_log)
    assert cat == "hallucination"
    assert any("ungrounded_entities" in r for r in reasons)

    # Disagreement classification
    disagree_log = {
        "fieldAgreements": {"emailExactMatch": False, "titleTokenF1": 0.5},
        "groundingFlags": {"allGrounded": True},
    }
    cat, reasons = classify_log_for_review(disagree_log)
    assert cat == "disagreement"
    assert "email_mismatch" in reasons
    assert any("title_f1_low" in r for r in reasons)

    # Agreement control classification
    agree_log = {
        "fieldAgreements": {"emailExactMatch": True, "titleTokenF1": 1.0, "skillsJaccard": 0.9},
        "groundingFlags": {"allGrounded": True},
    }
    cat, reasons = classify_log_for_review(agree_log)
    assert cat == "agreement_control"


def test_generate_stratified_sample_quotas():
    logs = []
    # 5 errors, 5 hallucinations, 10 disagreements, 20 controls
    for i in range(5):
        logs.append({"id": f"err_{i}", "errorFlags": ["timeout"], "isValidJson": True})
    for i in range(5):
        logs.append({"id": f"hal_{i}", "groundingFlags": {"allGrounded": False, "ungroundedEntities": [f"skill:Skill{i}"]}})
    for i in range(10):
        logs.append({"id": f"dis_{i}", "fieldAgreements": {"emailExactMatch": False, "titleTokenF1": 0.4}, "groundingFlags": {"allGrounded": True}})
    for i in range(20):
        logs.append({"id": f"agr_{i}", "fieldAgreements": {"emailExactMatch": True, "titleTokenF1": 1.0}, "groundingFlags": {"allGrounded": True}})

    sample = generate_stratified_sample(logs, target_sample_size=15, control_ratio=0.20)
    assert sample["actualSampleSize"] == 15
    # Should include errors and hallucinations first
    assert sample["distribution"]["error"] == 5
    assert sample["distribution"]["hallucination"] == 5
    assert sample["distribution"]["agreement_control"] >= 3


# ---------------------------------------------------------------------------
# 3. Labeling Interface & Storage Tests
# ---------------------------------------------------------------------------

def test_label_storage_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        labels_path = Path(tmp_dir) / "test_labels.json"
        data = load_review_labels(labels_path)
        assert data["labels"] == {}

        batch = [
            {
                "fileHash": "hash_abc",
                "overallJudgment": "slm_right",
                "fieldJudgments": {"candidateName": "both_right", "jobTitle": "slm_right"},
                "notes": "Preserved parentheses in title",
            },
            {
                "fileHash": "hash_xyz",
                "overallJudgment": "both_right",
                "fieldJudgments": {"candidateName": "both_right", "email": "both_right"},
                "notes": "Clean agreement",
            },
        ]

        applied = apply_batch_labels(batch, labels_path)
        assert applied == 2

        reloaded = load_review_labels(labels_path)
        assert len(reloaded["labels"]) == 2
        assert reloaded["labels"]["hash_abc"]["overallJudgment"] == "slm_right"


# ---------------------------------------------------------------------------
# 4. Promotion Gates Evaluation Tests
# ---------------------------------------------------------------------------

def test_gate_1_labeled_accuracy_pass():
    labels_store = {
        "labels": {
            f"item_{i}": {
                "fileHash": f"h_{i}",
                "fieldJudgments": {
                    "candidateName": "both_right",
                    "email": "slm_right" if i == 0 else "both_right",
                    "phone": "both_right",
                    "jobTitle": "slm_right" if i in (1, 2) else "both_right",
                    "totalExperienceYears": "both_right",
                },
                "notes": "fixed title truncation" if i in (1, 2) else "",
            }
            for i in range(6)
        }
    }
    g1 = evaluate_gate_1_labeled_accuracy(labels_store)
    assert g1["status"] == "PASS"
    assert g1["all_critical_fields_pass"] is True
    assert g1["bug_classes"]["title_truncation_preserved"]["count"] == 2


def test_gate_2_error_rate_thresholds():
    # Pass case
    pass_metrics = {"errors": {"totalErrorRatePct": 0.45, "invalidJsonRatePct": 0.1, "timeoutRatePct": 0.35}}
    g2_pass = evaluate_gate_2_error_rate(pass_metrics)
    assert g2_pass["status"] == "PASS"

    # Fail case
    fail_metrics = {"errors": {"totalErrorRatePct": 1.85, "invalidJsonRatePct": 1.2, "timeoutRatePct": 0.65}}
    g2_fail = evaluate_gate_2_error_rate(fail_metrics)
    assert g2_fail["status"] == "FAIL"


def test_gate_3_and_gate_4():
    # Memory safety: zero trips
    g3 = evaluate_gate_3_memory_safety(memory_guard_incidents=0)
    assert g3["status"] == "PASS"
    g3_bad = evaluate_gate_3_memory_safety(memory_guard_incidents=2)
    assert g3_bad["status"] == "FAIL"

    # Grounding: < 2%
    g4_pass = evaluate_gate_4_grounding({"grounding": {"groundingFailureRatePct": 0.8}})
    assert g4_pass["status"] == "PASS"
    g4_fail = evaluate_gate_4_grounding({"grounding": {"groundingFailureRatePct": 3.4}})
    assert g4_fail["status"] == "FAIL"


def test_full_promotion_gate_audit_verdict():
    audit = run_promotion_gate_audit()
    assert "gates" in audit
    assert "overallVerdict" in audit
    assert "gate1_labeled_accuracy" in audit["gates"]
    assert "gate2_error_rate" in audit["gates"]
    assert "gate3_memory_safety" in audit["gates"]
    assert "gate4_grounding_hallucination" in audit["gates"]
    assert "gate5_ranking_impact" in audit["gates"]


# ---------------------------------------------------------------------------
# 5. Ranking Impact Correlation & Regression Tests
# ---------------------------------------------------------------------------

def test_rank_correlation_math():
    # Identical rankings: rho = 1.0
    assert compute_spearman_rank_correlation([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    # Reversed rankings: rho = -1.0
    assert compute_spearman_rank_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    # Slight swap: high correlation
    rho = compute_spearman_rank_correlation([1, 2, 3, 4], [2, 1, 3, 4])
    assert rho >= 0.80

    # Pearson correlation
    assert compute_pearson_correlation([80.0, 90.0, 60.0], [80.0, 90.0, 60.0]) == 1.0


def test_ranking_impact_full_pipeline_audit():
    corpus = load_sample_resumes_corpus()
    assert len(corpus) >= 4

    target_jd = JobDescription(
        job_id="JD_TEST_BACKEND",
        title="Senior Backend & Cloud Engineer",
        raw_text="Required: 5+ years Go, Python, Kubernetes, AWS, Docker, PostgreSQL. BS in CS.",
        required_skills=["go", "python", "kubernetes", "docker", "postgresql"],
        preferred_skills=["aws", "grpc", "kafka", "redis"],
        mandatory_skills=["go", "kubernetes"],
        minimum_experience_years=5.0,
        education_requirements=["bachelor"],
    )

    audit = run_ranking_impact_audit(corpus, target_jd)
    assert audit["total_candidates"] == len(corpus)
    # Spearman rank correlation between legacy and SLM must remain high
    assert audit["spearman_rank_correlation"] >= 0.80
    assert audit["score_pearson_correlation"] >= 0.85
    assert audit["bracket_agreement_pct"] >= 75.0
    assert audit["mean_score_delta"] <= 8.0
