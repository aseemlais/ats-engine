"""
scripts/evaluate_shadow_metrics.py
==================================
Phase 4 Shadow Mode Analytics & Metric Evaluation Script.

Computes:
  1. Field-level agreement rates (email, phone, name F1, title F1, skills Jaccard, exp delta)
  2. Slice breakdown (PDF vs DOCX, page count, single vs two-column, length tiers)
  3. SLM invalid JSON and timeout error rates
  4. Validity-rate comparison (Legacy vs SLM: email, phone, name, dates)
  5. Grounding / Hallucination failure rate and ungrounded entities breakdown
  6. Latency distribution (p50, p95, min, max) and queue-wait impact
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_shadow_metrics")


def percentile(values: List[float], p: float) -> float:
    """Computes the p-th percentile (0 <= p <= 100) from a list of floats."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return round(d0 + d1, 2)


def fetch_shadow_logs_from_supabase(limit: int = 1000) -> List[Dict[str, Any]]:
    """Fetches recent shadow log rows from Supabase."""
    from app.storage.resume_storage import get_supabase_client, is_supabase_configured
    if not is_supabase_configured():
        logger.warning("Supabase is not configured; cannot fetch remote shadow logs.")
        return []

    client = get_supabase_client()
    if not client:
        return []

    from app.services.extraction_service import _get_table_ref
    table = _get_table_ref(client, "extraction_shadow_logs")
    res = table.select("*").order("createdAt", desc=True).limit(limit).execute()
    return res.data if res and res.data else []


def analyze_shadow_logs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyzes a list of shadow log records and generates comprehensive metrics."""
    if not logs:
        return {"error": "No shadow logs available to analyze.", "total_runs": 0}

    total_runs = len(logs)

    # 1. Field-Level Agreements
    email_matches = 0
    phone_matches = 0
    name_f1s: List[float] = []
    title_f1s: List[float] = []
    skills_jaccards: List[float] = []
    skills_precisions: List[float] = []
    skills_recalls: List[float] = []
    exp_deltas: List[float] = []
    edu_matches = 0
    work_matches = 0

    # 2. Validity Comparisons
    slm_email_valid = 0
    slm_phone_valid = 0
    slm_name_plausible = 0
    slm_dates_parse = 0

    leg_email_valid = 0
    leg_phone_valid = 0
    leg_name_plausible = 0
    leg_dates_parse = 0

    # 3. Errors & Grounding
    invalid_json_count = 0
    timeout_count = 0
    grounding_failures = 0
    ungrounded_entities_tally: Dict[str, int] = {}

    # 4. Latencies
    latencies: List[float] = []

    # 5. Slices
    slices: Dict[str, Dict[str, Any]] = {}

    for row in logs:
        # Harmonize field agreements
        agreements = row.get("fieldAgreements") or row.get("agreementScores") or {}
        if isinstance(agreements, str):
            try:
                agreements = json.loads(agreements)
            except Exception:
                agreements = {}

        if agreements.get("emailExactMatch"):
            email_matches += 1
        if agreements.get("phoneExactMatch"):
            phone_matches += 1

        name_f1s.append(float(agreements.get("nameTokenF1", 0.0)))
        title_f1s.append(float(agreements.get("titleTokenF1", 0.0)))
        skills_jaccards.append(float(agreements.get("skillsJaccard", 0.0)))
        skills_precisions.append(float(agreements.get("skillsPrecision", 0.0)))
        skills_recalls.append(float(agreements.get("skillsRecall", 0.0)))
        exp_deltas.append(float(agreements.get("expDeltaYears", 0.0)))

        if agreements.get("educationCountMatch"):
            edu_matches += 1
        if agreements.get("workExperienceCountMatch"):
            work_matches += 1

        # Validity Flags (SLM)
        validity = row.get("validityFlags") or {}
        if isinstance(validity, str):
            try:
                validity = json.loads(validity)
            except Exception:
                validity = {}

        if validity.get("emailValid", True):
            slm_email_valid += 1
        if validity.get("phoneValid", True):
            slm_phone_valid += 1
        if validity.get("namePlausible", True):
            slm_name_plausible += 1
        if validity.get("datesParse", True):
            slm_dates_parse += 1

        # Legacy validity check from legacyOutput if present
        leg_out = row.get("legacyOutput") or {}
        if isinstance(leg_out, str):
            try:
                leg_out = json.loads(leg_out)
            except Exception:
                leg_out = {}
        if leg_out:
            from app.services.shadow_evaluator import compute_validity_flags
            leg_val = compute_validity_flags(leg_out)
            if leg_val.get("emailValid"):
                leg_email_valid += 1
            if leg_val.get("phoneValid"):
                leg_phone_valid += 1
            if leg_val.get("namePlausible"):
                leg_name_plausible += 1
            if leg_val.get("datesParse"):
                leg_dates_parse += 1
        else:
            # Fallback assumption parity
            leg_email_valid += 1 if validity.get("emailValid") else 0
            leg_phone_valid += 1 if validity.get("phoneValid") else 0
            leg_name_plausible += 1 if validity.get("namePlausible") else 0
            leg_dates_parse += 1 if validity.get("datesParse") else 0

        # Errors
        errs = row.get("errorFlags") or row.get("discrepancies") or []
        if isinstance(errs, str):
            try:
                errs = json.loads(errs)
            except Exception:
                errs = [errs]

        is_valid_json = row.get("isValidJson")
        if is_valid_json is False or "json_invalid" in errs or "json_parse_error" in errs:
            invalid_json_count += 1
        if "timeout" in errs:
            timeout_count += 1

        # Grounding
        grounding = row.get("groundingFlags") or {}
        if isinstance(grounding, str):
            try:
                grounding = json.loads(grounding)
            except Exception:
                grounding = {}
        if grounding.get("allGrounded") is False:
            grounding_failures += 1
            for ent in grounding.get("ungroundedEntities", []):
                prefix = ent.split(":")[0] if ":" in ent else "entity"
                ungrounded_entities_tally[prefix] = ungrounded_entities_tally.get(prefix, 0) + 1

        # Latency
        lat = row.get("slmLatencyMs") or row.get("shadowDurationMs")
        if lat is not None:
            try:
                latencies.append(float(lat))
            except (ValueError, TypeError):
                pass

        # Slice Tracking
        doc_format = row.get("docFormat") or ("docx" if "docx" in str(row.get("filename", "")).lower() else "pdf")
        page_count = row.get("pageCount") or 1
        is_two_col = bool(row.get("isTwoColumn", False))
        char_len = int(row.get("charLength") or len(str(leg_out.get("resumeText") or "")))

        length_tier = "short (<2k)" if char_len < 2000 else ("medium (2k-5k)" if char_len <= 5000 else "long (>5k)")
        slice_key = f"{doc_format.upper()} | {page_count}p | {'2-Col' if is_two_col else '1-Col'} | {length_tier}"

        if slice_key not in slices:
            slices[slice_key] = {"count": 0, "title_f1s": [], "skills_jaccards": [], "latencies": []}
        slices[slice_key]["count"] += 1
        slices[slice_key]["title_f1s"].append(float(agreements.get("titleTokenF1", 0.0)))
        slices[slice_key]["skills_jaccards"].append(float(agreements.get("skillsJaccard", 0.0)))
        if lat is not None:
            try:
                slices[slice_key]["latencies"].append(float(lat))
            except Exception:
                pass

    # Aggregate Slices
    slice_breakdown = []
    for sk, sdata in slices.items():
        c = sdata["count"]
        slice_breakdown.append({
            "slice": sk,
            "sample_count": c,
            "avg_title_f1": round(sum(sdata["title_f1s"]) / c, 4) if c else 0.0,
            "avg_skills_jaccard": round(sum(sdata["skills_jaccards"]) / c, 4) if c else 0.0,
            "avg_latency_ms": round(sum(sdata["latencies"]) / len(sdata["latencies"]), 1) if sdata["latencies"] else 0.0,
        })

    def _avg(arr: List[float], decimals: int = 4) -> float:
        return round(sum(arr) / len(arr), decimals) if arr else 0.0

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalRuns": total_runs,
        "agreements": {
            "emailExactMatchPct": round((email_matches / total_runs) * 100, 2),
            "phoneExactMatchPct": round((phone_matches / total_runs) * 100, 2),
            "avgNameTokenF1": _avg(name_f1s, 4),
            "avgTitleTokenF1": _avg(title_f1s, 4),
            "avgSkillsJaccard": _avg(skills_jaccards, 4),
            "avgSkillsPrecision": _avg(skills_precisions, 4),
            "avgSkillsRecall": _avg(skills_recalls, 4),
            "avgExpDeltaYears": _avg(exp_deltas, 2),
            "educationCountMatchPct": round((edu_matches / total_runs) * 100, 2),
            "workExperienceCountMatchPct": round((work_matches / total_runs) * 100, 2),
        },
        "validityComparison": {
            "email": {
                "legacyValidPct": round((leg_email_valid / total_runs) * 100, 2),
                "slmValidPct": round((slm_email_valid / total_runs) * 100, 2),
            },
            "phone": {
                "legacyValidPct": round((leg_phone_valid / total_runs) * 100, 2),
                "slmValidPct": round((slm_phone_valid / total_runs) * 100, 2),
            },
            "name": {
                "legacyValidPct": round((leg_name_plausible / total_runs) * 100, 2),
                "slmValidPct": round((slm_name_plausible / total_runs) * 100, 2),
            },
            "dates": {
                "legacyValidPct": round((leg_dates_parse / total_runs) * 100, 2),
                "slmValidPct": round((slm_dates_parse / total_runs) * 100, 2),
            },
        },
        "errors": {
            "invalidJsonCount": invalid_json_count,
            "invalidJsonRatePct": round((invalid_json_count / total_runs) * 100, 3),
            "timeoutCount": timeout_count,
            "timeoutRatePct": round((timeout_count / total_runs) * 100, 3),
            "totalErrorRatePct": round(((invalid_json_count + timeout_count) / total_runs) * 100, 3),
        },
        "grounding": {
            "groundingFailureCount": grounding_failures,
            "groundingFailureRatePct": round((grounding_failures / total_runs) * 100, 3),
            "groundedRatePct": round(((total_runs - grounding_failures) / total_runs) * 100, 3),
            "ungroundedEntitiesBreakdown": ungrounded_entities_tally,
        },
        "latency": {
            "p50Ms": percentile(latencies, 50.0),
            "p95Ms": percentile(latencies, 95.0),
            "minMs": round(min(latencies), 1) if latencies else 0.0,
            "maxMs": round(max(latencies), 1) if latencies else 0.0,
            "meanMs": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        },
        "sliceBreakdown": slice_breakdown,
    }


def generate_markdown_report(metrics: Dict[str, Any]) -> str:
    """Formats metrics dictionary into a clean markdown evaluation report."""
    agr = metrics.get("agreements", {})
    val = metrics.get("validityComparison", {})
    err = metrics.get("errors", {})
    gro = metrics.get("grounding", {})
    lat = metrics.get("latency", {})
    slices = metrics.get("sliceBreakdown", [])

    md = [
        "# ATS Resume Extraction: Phase 4 Shadow Mode Evaluation Report",
        f"**Generated:** {metrics.get('generatedAt')} | **Total Shadow Evaluations:** {metrics.get('totalRuns')}",
        "",
        "## 1. Field-Level Agreement Rates (Legacy vs SLM)",
        "| Field | Metric Type | Score / Rate |",
        "| :--- | :--- | :--- |",
        f"| **Email** | Exact String Match | **{agr.get('emailExactMatchPct')}%** |",
        f"| **Phone** | Normalized Digits Match | **{agr.get('phoneExactMatchPct')}%** |",
        f"| **Candidate Name** | Token F1 | **{agr.get('avgNameTokenF1')}** |",
        f"| **Job Title** | Token F1 | **{agr.get('avgTitleTokenF1')}** |",
        f"| **Skills** | Jaccard Overlap | **{agr.get('avgSkillsJaccard')}** (Prec: {agr.get('avgSkillsPrecision')}, Rec: {agr.get('avgSkillsRecall')}) |",
        f"| **Total Experience** | Mean Absolute Delta | **{agr.get('avgExpDeltaYears')} years** |",
        f"| **Education Count** | Exact Match | **{agr.get('educationCountMatchPct')}%** |",
        f"| **Work Experience Count** | Exact Match | **{agr.get('workExperienceCountMatchPct')}%** |",
        "",
        "## 2. Validity-Rate Comparison (Legacy vs Fine-Tuned SLM)",
        "| Field Check | Legacy Extractor | Fine-Tuned SLM | Delta |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Valid RFC Email** | {val.get('email', {}).get('legacyValidPct')}% | {val.get('email', {}).get('slmValidPct')}% | {round(val.get('email', {}).get('slmValidPct', 0) - val.get('email', {}).get('legacyValidPct', 0), 2)}% |",
        f"| **Valid Phone (>= 7 digits)** | {val.get('phone', {}).get('legacyValidPct')}% | {val.get('phone', {}).get('slmValidPct')}% | {round(val.get('phone', {}).get('slmValidPct', 0) - val.get('phone', {}).get('legacyValidPct', 0), 2)}% |",
        f"| **Plausible Name (Non-blank/Valid)** | {val.get('name', {}).get('legacyValidPct')}% | {val.get('name', {}).get('slmValidPct')}% | {round(val.get('name', {}).get('slmValidPct', 0) - val.get('name', {}).get('legacyValidPct', 0), 2)}% |",
        f"| **Dates Parse Successfully** | {val.get('dates', {}).get('legacyValidPct')}% | {val.get('dates', {}).get('slmValidPct')}% | {round(val.get('dates', {}).get('slmValidPct', 0) - val.get('dates', {}).get('legacyValidPct', 0), 2)}% |",
        "",
        "## 3. Reliability, Grounding & Error Rates",
        "| Quality Gate | Value | Target Status |",
        "| :--- | :--- | :--- |",
        f"| **Invalid JSON Rate** | {err.get('invalidJsonRatePct')}% ({err.get('invalidJsonCount')} / {metrics.get('totalRuns')}) | {'PASS (<1%)' if err.get('invalidJsonRatePct', 0) < 1.0 else 'WARN'} |",
        f"| **Upstream Timeout Rate** | {err.get('timeoutRatePct')}% ({err.get('timeoutCount')} / {metrics.get('totalRuns')}) | {'PASS (<1%)' if err.get('timeoutRatePct', 0) < 1.0 else 'WARN'} |",
        f"| **Combined Error Rate** | {err.get('totalErrorRatePct')}% | {'PASS (<1%)' if err.get('totalErrorRatePct', 0) < 1.0 else 'FAIL'} |",
        f"| **Grounding Failure Rate** | {gro.get('groundingFailureRatePct')}% | {'PASS (<2%)' if gro.get('groundingFailureRatePct', 0) < 2.0 else 'WARN'} |",
        f"| **Overall Grounded Extractions** | {gro.get('groundedRatePct')}% | Supported by Resume Text |",
        "",
        "## 4. Latency Profile (Single Free-Tier vCPU Worker)",
        f"- **Median Latency (p50):** {lat.get('p50Ms')} ms",
        f"- **95th Percentile (p95):** {lat.get('p95Ms')} ms",
        f"- **Mean Latency:** {lat.get('meanMs')} ms",
        f"- **Range:** {lat.get('minMs')} ms – {lat.get('maxMs')} ms",
        "",
        "## 5. Slice Breakdown (Format, Layout & Length)",
        "| Slice | Samples | Title F1 | Skills Jaccard | Avg Latency |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for s in slices:
        md.append(f"| {s['slice']} | {s['sample_count']} | {s['avg_title_f1']} | {s['avg_skills_jaccard']} | {s['avg_latency_ms']} ms |")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ATS Shadow Mode extraction metrics.")
    parser.add_argument("--from-file", type=str, help="Path to local JSON file of shadow logs.")
    parser.add_argument("--limit", type=int, default=500, help="Max records to pull from Supabase.")
    parser.add_argument("--export-report", action="store_true", default=True, help="Save report to reports/ dir.")
    args = parser.parse_args()

    logs = []
    if args.from_file and os.path.exists(args.from_file):
        logger.info("Loading shadow logs from local file: %s", args.from_file)
        with open(args.from_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logger.info("Fetching shadow logs from Supabase (limit=%d)...", args.limit)
        logs = fetch_shadow_logs_from_supabase(limit=args.limit)

    if not logs:
        logger.warning("No logs found. Generating synthetic shadow evaluation fixture for demonstration/testing.")
        from app.services.shadow_evaluator import (
            compute_field_agreements,
            compute_validity_flags,
            compute_grounding_flags,
        )
        sample_leg = {
            "candidateName": "David Chen",
            "email": "david.chen@example.com",
            "phone": "+1-415-555-0199",
            "jobTitle": "Senior Distributed Systems Engineer",
            "skills": ["Go", "Kubernetes", "gRPC", "Docker", "PostgreSQL"],
            "totalExperienceYears": 6.5,
            "education": [{"degree": "B.S. Computer Science"}],
            "workExperience": [{"title": "Senior Distributed Systems Engineer", "company": "CloudTech", "durationYears": 4.0}],
            "resumeText": "David Chen | david.chen@example.com | (415) 555-0199\nCloudTech Senior Distributed Systems Engineer\nSkills: Go, Kubernetes, gRPC, Docker, PostgreSQL",
        }
        sample_slm = {
            "candidateName": "David Chen",
            "email": "david.chen@example.com",
            "phone": "(415) 555-0199",
            "jobTitle": "Senior Distributed Systems Engineer (Go/K8s)",
            "skills": ["Go", "Kubernetes", "gRPC", "Docker", "PostgreSQL", "Kafka"],
            "totalExperienceYears": 6.5,
            "education": [{"degree": "B.S. Computer Science"}],
            "workExperience": [{"title": "Senior Distributed Systems Engineer", "company": "CloudTech", "durationYears": 4.0}],
        }
        agreements = compute_field_agreements(sample_leg, sample_slm)
        validity = compute_validity_flags(sample_slm)
        grounding = compute_grounding_flags(sample_slm, sample_leg["resumeText"])
        logs = [{
            "fileHash": "test_hash_12345",
            "legacyVersion": "ats-legacy:heuristic-v1",
            "slmVersion": "unsloth-qwen2.5-1.5b-q4_k_m",
            "legacyOutput": sample_leg,
            "slmOutput": sample_slm,
            "fieldAgreements": agreements,
            "validityFlags": validity,
            "groundingFlags": grounding,
            "slmLatencyMs": 1420.5,
            "docFormat": "pdf",
            "pageCount": 2,
            "isTwoColumn": False,
            "charLength": 2400,
            "isValidJson": True,
            "errorFlags": [],
        }]

    metrics = analyze_shadow_logs(logs)
    report_md = generate_markdown_report(metrics)
    print("\n" + report_md + "\n")

    if args.export_report:
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        rep_path = reports_dir / "shadow_evaluation_report.md"
        json_path = reports_dir / "shadow_evaluation_summary.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved evaluation report to %s and %s", rep_path, json_path)


if __name__ == "__main__":
    main()
