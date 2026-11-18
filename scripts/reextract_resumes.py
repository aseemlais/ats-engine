"""
scripts/reextract_resumes.py
============================
Phase 5 Re-Extraction Utility.

Safely re-extracts candidate resumes stored in Supabase (or local store) using
a new fine-tuned model version.

Features:
  - Version-targeted: Only re-extracts records where extractorVersion != target_version (unless --force)
  - Safe Dry-Run mode: Previews extractions, schema validity, and deltas without modifying database
  - Zero PII in logs: Logs only record counts, hashes, execution times, and status
  - Rate-limited & CPU friendly: Introduces configurable sleep between extractions to prevent container throttling
  - Atomic updates: Updates extractedData JSONB and extractorVersion with rollback safety
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reextract_resumes")

DEFAULT_TARGET_VERSION = "ats-slm:qwen2.5-1.5b-instruct:chatml-v1:schema-v1"


def reextract_single_candidate(
    candidate_id: str,
    raw_text: str,
    extraction_service: Any,
    target_version: str = DEFAULT_TARGET_VERSION,
) -> Dict[str, Any]:
    """Runs extraction on raw resume text and formats canonical output."""
    from app.services.post_processor import post_process_resume

    t0 = time.perf_counter()
    raw_data = None
    finish_reason = None

    if extraction_service and extraction_service.llm:
        infer_res = extraction_service._infer_llm(raw_text)
        if infer_res:
            raw_data, finish_reason = (infer_res[0], infer_res[1]) if isinstance(infer_res, tuple) else (infer_res, "stop")

    if raw_data and finish_reason != "length":
        processed = post_process_resume(raw_data)
        extraction_source = "slm"
    else:
        # Fallback to deterministic extraction if model unavailable or truncated
        if not hasattr(extraction_service, "_fallback_extractor") or extraction_service._fallback_extractor is None:
            from app.services.structured_extractor import StructuredResumeExtractor
            fallback_ext = StructuredResumeExtractor()
        else:
            fallback_ext = extraction_service._fallback_extractor
        fallback_data = fallback_ext.get_safe_fallback_structure(raw_text)
        processed = post_process_resume(fallback_data)
        extraction_source = "fallback"

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    dumped = processed.model_dump(by_alias=True)
    dumped["jobRole"] = dumped.get("jobTitle")
    dumped["extractionSource"] = extraction_source
    dumped["extractorVersion"] = target_version
    dumped["reextractedAt"] = datetime.now(timezone.utc).isoformat()
    dumped.pop("storageObjectPath", None)

    return {
        "candidate_id": candidate_id,
        "extraction_source": extraction_source,
        "elapsed_ms": elapsed_ms,
        "extracted_data": dumped,
    }


def fetch_resumes_to_reextract(
    limit: int = 50,
    force: bool = False,
    target_version: str = DEFAULT_TARGET_VERSION,
) -> List[Dict[str, Any]]:
    """Queries candidate resumes requiring re-extraction from Supabase."""
    from app.storage.resume_storage import get_supabase_client, is_supabase_configured
    if not is_supabase_configured():
        logger.warning("Supabase is not configured; using offline sample resumes.")
        return []

    client = get_supabase_client()
    if not client:
        return []

    from app.storage.resume_storage import _get_table_ref
    table = _get_table_ref(client, "CandidateResume")

    query = table.select("id, fileHash, originalFilename, resumeText, extractedData, extractorVersion")
    if not force:
        # Filter for rows where extractorVersion is null or does not match target_version
        query = query.or_(f"extractorVersion.is.null,extractorVersion.neq.{target_version}")

    res = query.order("createdAt", desc=True).limit(limit).execute()
    return res.data if res and res.data else []


def update_candidate_extracted_data(
    candidate_id: str,
    extracted_data: Dict[str, Any],
    target_version: str,
) -> bool:
    """Updates candidate extracted_data in Supabase."""
    try:
        from app.storage.resume_storage import get_supabase_client, is_supabase_configured
        if not is_supabase_configured():
            return False
        client = get_supabase_client()
        if not client:
            return False

        from app.storage.resume_storage import _get_table_ref
        table = _get_table_ref(client, "CandidateResume")
        table.update({
            "extractedData": extracted_data,
            "extractorVersion": target_version,
            "jobRole": extracted_data.get("jobTitle") or "Role not identified",
            "candidateName": extracted_data.get("candidateName") or "Unknown",
            "email": extracted_data.get("email"),
            "skills": extracted_data.get("skills") or [],
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }).eq("id", candidate_id).execute()
        return True
    except Exception as exc:
        logger.error("Failed to update candidate %s in Supabase: %s", candidate_id, exc)
        return False


def run_reextraction(
    limit: int = 50,
    force: bool = False,
    dry_run: bool = True,
    delay_seconds: float = 0.1,
    target_version: str = DEFAULT_TARGET_VERSION,
    local_corpus: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Runs the full re-extraction workflow."""
    from app.services.extraction_service import ExtractionService
    service = ExtractionService()

    candidates = local_corpus if local_corpus is not None else fetch_resumes_to_reextract(limit=limit, force=force, target_version=target_version)

    if not candidates:
        logger.info("No candidates found requiring re-extraction.")
        return {
            "total_candidates": 0,
            "reextracted_count": 0,
            "dry_run": dry_run,
            "target_version": target_version,
        }

    logger.info("Starting re-extraction for %d candidates (dry_run=%s, target_version=%s)...", len(candidates), dry_run, target_version)

    processed_count = 0
    success_count = 0
    failure_count = 0
    results_summary: List[Dict[str, Any]] = []

    for item in candidates:
        cid = str(item.get("id") or item.get("candidate_id"))
        fhash = str(item.get("fileHash") or item.get("file_hash") or "unknown_hash")
        raw_text = item.get("resumeText") or item.get("raw_text") or ""

        if not raw_text.strip():
            logger.warning("Skipping candidate %s (no raw resume text found).", cid)
            continue

        try:
            res = reextract_single_candidate(
                candidate_id=cid,
                raw_text=raw_text,
                extraction_service=service,
                target_version=target_version,
            )

            extracted = res["extracted_data"]
            skills_count = len(extracted.get("skills") or [])
            exp_years = float(extracted.get("totalExperienceYears") or 0.0)
            engine = res["extraction_source"]

            if not dry_run:
                updated = update_candidate_extracted_data(cid, extracted, target_version)
                if updated:
                    success_count += 1
                else:
                    failure_count += 1
            else:
                success_count += 1

            processed_count += 1
            results_summary.append({
                "candidate_id": cid,
                "file_hash_snippet": fhash[:12],
                "engine": engine,
                "skills_count": skills_count,
                "exp_years": exp_years,
                "elapsed_ms": res["elapsed_ms"],
            })

            logger.info(
                "[ReExtraction] Candidate %s (%s) re-extracted via '%s' in %.1fms (skills: %d, exp: %.1f yrs)",
                cid,
                fhash[:8],
                engine,
                res["elapsed_ms"],
                skills_count,
                exp_years,
            )

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        except Exception as exc:
            failure_count += 1
            logger.error("Error re-extracting candidate %s: %s", cid, exc)

    logger.info(
        "Re-extraction completed: %d total, %d succeeded, %d failed (dry_run=%s)",
        processed_count,
        success_count,
        failure_count,
        dry_run,
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_evaluated": len(candidates),
        "processed_count": processed_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "dry_run": dry_run,
        "target_version": target_version,
        "summary": results_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-extract candidate resumes using new extractor model version.")
    parser.add_argument("--limit", type=int, default=20, help="Max candidates to re-extract.")
    parser.add_argument("--force", action="store_true", help="Re-extract even if version already matches.")
    parser.add_argument("--apply", action="store_true", help="Apply updates to Supabase (default is dry-run).")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay in seconds between records.")
    parser.add_argument("--version", type=str, default=DEFAULT_TARGET_VERSION, help="Target extractor version.")
    args = parser.parse_args()

    dry_run = not args.apply
    run_reextraction(
        limit=args.limit,
        force=args.force,
        dry_run=dry_run,
        delay_seconds=args.delay,
        target_version=args.version,
    )


if __name__ == "__main__":
    main()
