"""
backfill_extracted_data.py
==========================
Backfills extractedData, extractorVersion, and extractionSource into CandidateResume rows
that either lack extractedData or were extracted with an older extractorVersion.

Safety & Guardrails:
  - Dry-run by default (--dry-run).
  - Prints a before/after diff for sample rows.
  - Requires --apply to execute writes.
  - Refuses to process all rows unless --all is explicitly passed with --apply.
  - Supports --limit N (default: 10).

Usage:
  # Dry-run over 5 rows:
  python scripts/backfill_extracted_data.py --limit 5

  # Apply backfill to 10 rows:
  python scripts/backfill_extracted_data.py --limit 10 --apply

  # Apply backfill to all rows (requires explicit --all):
  python scripts/backfill_extracted_data.py --all --apply
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.storage.supabase_client import get_supabase_client, is_supabase_configured
from app.storage.resume_storage import _get_table_ref
from app.services.extraction_service import ExtractionService
from app.services.post_processor import post_process_resume
from app.services.structured_extractor import StructuredResumeExtractor

logger = logging.getLogger("backfill_extracted_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_backfill(limit: Optional[int] = 10, process_all: bool = False, apply: bool = False):
    if not is_supabase_configured():
        logger.error("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY.")
        sys.exit(1)

    if not limit and not process_all:
        logger.error("Refusing to process all records without explicit --all flag. Pass --limit N or --all.")
        sys.exit(1)

    client = get_supabase_client()
    table = _get_table_ref(client, "CandidateResume")

    # Target version
    service = ExtractionService()
    target_version = service.current_extractor_version
    logger.info("Target extractorVersion: '%s'", target_version)

    # Query rows needing backfill (extractedData is null or version != target_version)
    query = table.select("id, fileHash, originalFilename, candidateName, jobRole, resumeText, extractedData, extractorVersion")
    if limit and not process_all:
        query = query.limit(limit)

    res = query.execute()
    rows = res.data if res and res.data else []

    candidate_rows = [
        r for r in rows 
        if not r.get("extractedData") or r.get("extractorVersion") != target_version
    ]

    logger.info("Found %d rows needing backfill (queried %d rows).", len(candidate_rows), len(rows))

    if not candidate_rows:
        logger.info("No rows require backfill.")
        return

    legacy_extractor = StructuredResumeExtractor()
    backfilled_count = 0

    for idx, row in enumerate(candidate_rows, start=1):
        file_hash = row.get("fileHash")
        resume_text = row.get("resumeText") or ""
        old_version = row.get("extractorVersion")

        if not resume_text.strip():
            logger.warning("[%d/%d] Skipping candidate %s: resumeText is empty.", idx, len(candidate_rows), file_hash[:12])
            continue

        # Extract structured data
        raw_structure = legacy_extractor.get_safe_fallback_structure(resume_text)
        processed = post_process_resume(raw_structure)
        extracted_data = processed.model_dump(by_alias=True)
        extracted_data["jobRole"] = extracted_data.get("jobTitle")
        extracted_data["extractionSource"] = "backfill"
        extracted_data["extractorVersion"] = target_version

        # Show diff for first few samples
        if idx <= 3 or not apply:
            logger.info("--- [Sample %d] Diff for hash %s (%s) ---", idx, file_hash[:12], row.get("originalFilename"))
            logger.info("  Before: extractedData=%s, extractorVersion=%s", "PRESENT" if row.get("extractedData") else "NULL", old_version)
            logger.info("  After : Name='%s', Role='%s', Skills=%d, Exp=%.1f yrs, Version='%s'",
                        extracted_data.get("candidateName"), extracted_data.get("jobTitle"),
                        len(extracted_data.get("skills", [])), extracted_data.get("totalExperienceYears", 0.0), target_version)

        if apply:
            update_payload = {
                "extractedData": extracted_data,
                "extractorVersion": target_version,
                "extractionSource": "backfill",
                "candidateName": extracted_data.get("candidateName") or row.get("candidateName"),
                "jobRole": extracted_data.get("jobTitle") or row.get("jobRole"),
            }
            table.update(update_payload).eq("fileHash", file_hash).execute()
            backfilled_count += 1

    if apply:
        logger.info("Successfully backfilled %d candidate records.", backfilled_count)
    else:
        logger.info("[DRY-RUN] Processed %d records in preview mode. Run with --apply to commit changes.", len(candidate_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill extractedData into CandidateResume")
    parser.add_argument("--limit", type=int, default=10, help="Max records to process (default: 10)")
    parser.add_argument("--all", action="store_true", help="Process all records needing backfill (requires --apply)")
    parser.add_argument("--apply", action="store_true", help="Commit writes to Supabase (default: dry-run only)")
    args = parser.parse_args()

    limit = None if args.all else args.limit
    run_backfill(limit=limit, process_all=args.all, apply=args.apply)
