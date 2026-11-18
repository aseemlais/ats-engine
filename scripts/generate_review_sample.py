"""
scripts/generate_review_sample.py
=================================
Phase 4 Stratified Review Sample Generator.

Generates a weekly review sample from shadow logs:
  - Prioritizes high disagreements (email/phone mismatch, title F1 < 0.75, skills Jaccard < 0.60, exp delta >= 1.0)
  - Prioritizes hallucination / ungrounded claims (allGrounded == False)
  - Prioritizes errors (invalid JSON, timeout, length limit)
  - Includes a random control slice of agreements (20%) to confirm consistency
  - Outputs a review queue file to data/review_samples/review_sample_<timestamp>.json
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_review_sample")


def classify_log_for_review(row: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Evaluates a shadow log record and returns (category, reasons).
    Categories: 'error', 'hallucination', 'disagreement', 'agreement_control'
    """
    reasons: List[str] = []

    # 1. Error check
    errs = row.get("errorFlags") or row.get("discrepancies") or []
    if isinstance(errs, str):
        try:
            errs = json.loads(errs)
        except Exception:
            errs = [errs]
    is_valid_json = row.get("isValidJson")
    if is_valid_json is False or any(e in ("json_invalid", "json_parse_error") for e in errs):
        reasons.append("invalid_json_format")
    if "timeout" in errs:
        reasons.append("upstream_timeout")
    if "finish_reason_length" in errs:
        reasons.append("context_length_truncation")
    if reasons:
        return "error", reasons

    # 2. Grounding check
    grounding = row.get("groundingFlags") or {}
    if isinstance(grounding, str):
        try:
            grounding = json.loads(grounding)
        except Exception:
            grounding = {}
    if grounding.get("allGrounded") is False:
        ungrounded = grounding.get("ungroundedEntities", [])
        reasons.append(f"ungrounded_entities({len(ungrounded)}): {', '.join(ungrounded[:3])}")
        return "hallucination", reasons

    # 3. Disagreement check
    agreements = row.get("fieldAgreements") or row.get("agreementScores") or {}
    if isinstance(agreements, str):
        try:
            agreements = json.loads(agreements)
        except Exception:
            agreements = {}

    if agreements.get("emailExactMatch") is False:
        reasons.append("email_mismatch")
    if agreements.get("phoneExactMatch") is False:
        reasons.append("phone_mismatch")

    title_f1 = float(agreements.get("titleTokenF1", 1.0))
    if title_f1 < 0.75:
        reasons.append(f"title_f1_low({title_f1:.2f})")

    skills_jaccard = float(agreements.get("skillsJaccard", 1.0))
    if skills_jaccard < 0.60:
        reasons.append(f"skills_jaccard_low({skills_jaccard:.2f})")

    exp_delta = float(agreements.get("expDeltaYears", 0.0))
    if exp_delta >= 1.0:
        reasons.append(f"exp_delta_high({exp_delta:.1f}y)")

    if agreements.get("educationCountMatch") is False:
        reasons.append("education_count_mismatch")

    if reasons:
        return "disagreement", reasons

    return "agreement_control", ["high_agreement_control"]


def generate_stratified_sample(
    logs: List[Dict[str, Any]],
    target_sample_size: int = 30,
    control_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates a stratified review sample prioritizing errors, hallucinations, and disagreements."""
    random.seed(seed)

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "error": [],
        "hallucination": [],
        "disagreement": [],
        "agreement_control": [],
    }

    for row in logs:
        cat, reasons = classify_log_for_review(row)
        item = dict(row)
        item["sampleCategory"] = cat
        item["disagreementReasons"] = reasons
        buckets[cat].append(item)

    # Calculate quota allocation
    control_quota = max(1, int(target_sample_size * control_ratio))
    disagreement_quota = target_sample_size - control_quota

    selected: List[Dict[str, Any]] = []

    # Priority 1: Errors (all available up to quota)
    selected.extend(buckets["error"][:disagreement_quota])
    remaining = disagreement_quota - len(selected)

    # Priority 2: Hallucinations
    if remaining > 0:
        selected.extend(buckets["hallucination"][:remaining])
        remaining = disagreement_quota - len(selected)

    # Priority 3: Disagreements
    if remaining > 0:
        disagreements = buckets["disagreement"]
        random.shuffle(disagreements)
        selected.extend(disagreements[:remaining])
        remaining = disagreement_quota - len(selected)

    # Priority 4: Control slice of agreements
    controls = buckets["agreement_control"]
    random.shuffle(controls)
    selected.extend(controls[:control_quota])

    # If still below target, fill with remaining controls
    if len(selected) < target_sample_size and len(controls) > control_quota:
        extra_needed = target_sample_size - len(selected)
        selected.extend(controls[control_quota:control_quota + extra_needed])

    # Strip excessive raw fields for cleaner review
    review_queue: List[Dict[str, Any]] = []
    for s in selected:
        leg_out = s.get("legacyOutput") or {}
        slm_out = s.get("slmOutput") or {}
        review_queue.append({
            "id": s.get("id"),
            "fileHash": s.get("fileHash"),
            "sampleCategory": s.get("sampleCategory"),
            "disagreementReasons": s.get("disagreementReasons"),
            "docFormat": s.get("docFormat", "pdf"),
            "pageCount": s.get("pageCount", 1),
            "createdAt": s.get("createdAt"),
            "legacyOutput": {
                "candidateName": leg_out.get("candidateName"),
                "email": leg_out.get("email"),
                "phone": leg_out.get("phone"),
                "jobTitle": leg_out.get("jobTitle"),
                "skills": leg_out.get("skills", []),
                "totalExperienceYears": leg_out.get("totalExperienceYears", 0.0),
                "education": leg_out.get("education", []),
                "workExperience": leg_out.get("workExperience", []),
            },
            "slmOutput": {
                "candidateName": slm_out.get("candidateName"),
                "email": slm_out.get("email"),
                "phone": slm_out.get("phone"),
                "jobTitle": slm_out.get("jobTitle"),
                "skills": slm_out.get("skills", []),
                "totalExperienceYears": slm_out.get("totalExperienceYears", 0.0),
                "education": slm_out.get("education", []),
                "workExperience": slm_out.get("workExperience", []),
            },
            "groundingFlags": s.get("groundingFlags", {}),
            "validityFlags": s.get("validityFlags", {}),
            "fieldAgreements": s.get("fieldAgreements", {}),
        })

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "totalSourceLogs": len(logs),
        "targetSampleSize": target_sample_size,
        "actualSampleSize": len(review_queue),
        "distribution": {
            "error": len([x for x in review_queue if x["sampleCategory"] == "error"]),
            "hallucination": len([x for x in review_queue if x["sampleCategory"] == "hallucination"]),
            "disagreement": len([x for x in review_queue if x["sampleCategory"] == "disagreement"]),
            "agreement_control": len([x for x in review_queue if x["sampleCategory"] == "agreement_control"]),
        },
        "reviewQueue": review_queue,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate stratified review sample from shadow logs.")
    parser.add_argument("--from-file", type=str, help="Path to input shadow logs JSON file.")
    parser.add_argument("--sample-size", type=int, default=25, help="Number of items to sample for weekly review.")
    parser.add_argument("--output-dir", type=str, default="data/review_samples", help="Output directory for sample.")
    args = parser.parse_args()

    logs = []
    if args.from_file and os.path.exists(args.from_file):
        with open(args.from_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        from scripts.evaluate_shadow_metrics import fetch_shadow_logs_from_supabase
        logs = fetch_shadow_logs_from_supabase(limit=500)

    if not logs:
        logger.info("No remote logs found. Generating representative synthetic shadow logs for sampling test.")
        # Create diverse sample of 35 items
        for i in range(35):
            is_disagree = (i % 3 == 0)
            is_halluc = (i == 7)
            is_err = (i == 13)
            leg_name = f"Candidate {i}"
            slm_name = leg_name if not is_disagree else f"Dr. {leg_name} PhD"
            leg_title = "Backend Engineer"
            slm_title = "Senior Backend Engineer (Go/K8s)" if is_disagree else "Backend Engineer"
            logs.append({
                "id": f"sample-uuid-{i}",
                "fileHash": f"hash_{i:04d}_{i*997}",
                "docFormat": "pdf" if i % 2 == 0 else "docx",
                "pageCount": 1 + (i % 3),
                "isTwoColumn": (i % 4 == 0),
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "legacyOutput": {
                    "candidateName": leg_name,
                    "email": f"cand{i}@example.com" if not (i % 5 == 0) else "—",
                    "phone": f"555-01{i:02d}",
                    "jobTitle": leg_title,
                    "skills": ["Python", "Docker"] if not is_disagree else ["Python"],
                    "totalExperienceYears": 3.0 + i,
                },
                "slmOutput": {
                    "candidateName": slm_name,
                    "email": f"cand{i}@example.com",
                    "phone": f"+1-555-01{i:02d}",
                    "jobTitle": slm_title,
                    "skills": ["Python", "Docker", "Kubernetes"] if is_disagree else ["Python", "Docker"],
                    "totalExperienceYears": 3.0 + i,
                },
                "fieldAgreements": {
                    "emailExactMatch": not (i % 5 == 0),
                    "phoneExactMatch": True,
                    "titleTokenF1": 0.50 if is_disagree else 1.0,
                    "skillsJaccard": 0.50 if is_disagree else 1.0,
                    "expDeltaYears": 0.0,
                },
                "groundingFlags": {
                    "allGrounded": not is_halluc,
                    "ungroundedEntities": ["skill:QuantumComputing"] if is_halluc else [],
                },
                "errorFlags": ["json_invalid"] if is_err else [],
                "isValidJson": not is_err,
                "validityFlags": {
                    "emailValid": True,
                    "phoneValid": True,
                    "namePlausible": True,
                    "datesParse": True,
                },
            })

    sample_res = generate_stratified_sample(logs, target_sample_size=args.sample_size)
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"review_sample_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample_res, f, indent=2)

    logger.info("Generated stratified sample of %d items to %s", sample_res["actualSampleSize"], out_path)
    logger.info("Sample distribution: %s", sample_res["distribution"])


if __name__ == "__main__":
    main()
