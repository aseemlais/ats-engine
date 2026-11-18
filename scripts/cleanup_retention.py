"""
cleanup_retention.py
====================
Standalone retention cleanup script for ExtractionJobs table.
Deletes extraction job records older than --days (default: 30 days) to prevent PII accumulation.

Usage:
  # Dry-run (default: checks and prints count without deleting)
  python scripts/cleanup_retention.py --days 30

  # Apply deletion
  python scripts/cleanup_retention.py --days 30 --apply
"""

import argparse
import datetime
import logging
import os
import sys

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.storage.supabase_client import get_supabase_client, is_supabase_configured
from app.storage.resume_storage import _get_table_ref

logger = logging.getLogger("retention_cleanup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_cleanup(days: int = 30, apply: bool = False):
    if not is_supabase_configured():
        logger.error("Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY.")
        sys.exit(1)

    client = get_supabase_client()
    cutoff_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    cutoff_iso = cutoff_dt.isoformat()

    logger.info("Evaluating jobs older than %d days (cutoff: %s)...", days, cutoff_iso)

    table = _get_table_ref(client, "ExtractionJobs")

    # Count eligible jobs
    res = table.select("jobId, createdAt, status").lt("createdAt", cutoff_iso).execute()
    eligible_count = len(res.data) if res and res.data else 0

    logger.info("Found %d eligible job records for retention deletion.", eligible_count)

    if not apply:
        logger.info("[DRY-RUN] No records were deleted. Run with --apply to execute deletion.")
        return

    if eligible_count == 0:
        logger.info("No records to delete.")
        return

    del_res = table.delete().lt("createdAt", cutoff_iso).execute()
    logger.info("Successfully deleted %d expired extraction job records.", eligible_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up expired ExtractionJobs records")
    parser.add_argument("--days", type=int, default=30, help="Retention window in days (default: 30)")
    parser.add_argument("--apply", action="store_true", help="Actually execute deletion (default: dry-run only)")
    args = parser.parse_args()

    run_cleanup(days=args.days, apply=args.apply)
