"""
scripts/label_disagreements.py
==============================
Phase 4 Minimal Review Labeling Interface.

Allows a reviewer to mark per-field judgments on extraction disagreements:
  - legacy_right: Legacy extractor was correct
  - slm_right: Fine-tuned SLM extractor was correct
  - both_right: Both are acceptable representations
  - both_wrong: Neither extracted the field properly
  - notes: Freeform reviewer notes

Stores labels into:
  - Local dataset: data/review_labels.json
  - Remote Supabase: updates `reviewLabel`, `reviewerNotes`, and `labeledAt` in `extraction_shadow_logs`
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("label_disagreements")

CHOICE_MAP = {
    "1": "legacy_right",
    "2": "slm_right",
    "3": "both_right",
    "4": "both_wrong",
    "l": "legacy_right",
    "s": "slm_right",
    "b": "both_right",
    "w": "both_wrong",
}

FIELDS_TO_LABEL = [
    ("candidateName", "Candidate Name"),
    ("email", "Email Address"),
    ("phone", "Phone Number"),
    ("jobTitle", "Current Job Title"),
    ("skills", "Technical Skills"),
    ("totalExperienceYears", "Total Experience Years"),
]


def load_review_labels(file_path: Path) -> Dict[str, Any]:
    """Loads existing labels from local storage."""
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to parse existing labels file %s: %s", file_path, e)
    return {"updatedAt": None, "labels": {}}


def save_review_labels(file_path: Path, data: Dict[str, Any]) -> None:
    """Saves updated labels dictionary to local storage."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data["updatedAt"] = datetime.now(timezone.utc).isoformat()
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def update_supabase_shadow_label(file_hash: str, label_data: Dict[str, Any]) -> bool:
    """Updates review label and notes in Supabase extraction_shadow_logs."""
    try:
        from app.storage.resume_storage import get_supabase_client, is_supabase_configured
        if not is_supabase_configured():
            return False
        client = get_supabase_client()
        if not client:
            return False

        overall = label_data.get("overallJudgment", "unspecified")
        notes = label_data.get("notes", "")

        from app.services.extraction_service import _get_table_ref
        table = _get_table_ref(client, "extraction_shadow_logs")
        table.update({
            "reviewLabel": overall,
            "reviewerNotes": f"Field judgments: {json.dumps(label_data.get('fieldJudgments', {}))} | Notes: {notes}",
            "labeledAt": datetime.now(timezone.utc).isoformat(),
        }).eq("fileHash", file_hash).execute()
        return True
    except Exception as exc:
        logger.warning("Failed to update Supabase shadow label for %s: %s", file_hash, exc)
        return False


def run_interactive_labeling(sample_file: Path, labels_file: Path) -> None:
    """Runs interactive terminal labeling loop."""
    if not sample_file.exists():
        logger.error("Sample file does not exist: %s", sample_file)
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    queue = sample_data.get("reviewQueue", [])
    if not queue:
        logger.info("No items in review queue.")
        return

    existing_store = load_review_labels(labels_file)
    labels = existing_store.get("labels", {})

    print("\n" + "=" * 70)
    print("  ATS RESUME EXTRACTION - HUMAN REVIEW LABELING TOOL")
    print("  Controls: [1] legacy_right | [2] slm_right | [3] both_right | [4] both_wrong")
    print("  Type 'skip' to advance, 'q' to save and exit.")
    print("=" * 70 + "\n")

    labeled_count = 0
    for idx, item in enumerate(queue):
        fhash = item.get("fileHash", f"item_{idx}")
        if fhash in labels:
            continue

        print(f"\n--- Item {idx + 1} / {len(queue)} [Hash: {fhash[:12]}] ---")
        print(f"Category: {item.get('sampleCategory')} | Reasons: {item.get('disagreementReasons')}")

        leg = item.get("legacyOutput", {})
        slm = item.get("slmOutput", {})

        field_judgments: Dict[str, str] = {}
        skip_item = False

        for field_key, field_name in FIELDS_TO_LABEL:
            v_leg = leg.get(field_key)
            v_slm = slm.get(field_key)

            print(f"\n[{field_name}]")
            print(f"  Legacy: {v_leg}")
            print(f"  SLM:    {v_slm}")

            while True:
                resp = input("  Choice [1=Leg, 2=SLM, 3=Both_OK, 4=Both_Bad, skip, q]: ").strip().lower()
                if resp == "q":
                    existing_store["labels"] = labels
                    save_review_labels(labels_file, existing_store)
                    print(f"\nSaved {labeled_count} new labels. Goodbye!")
                    return
                if resp == "skip":
                    skip_item = True
                    break
                if resp in CHOICE_MAP:
                    field_judgments[field_key] = CHOICE_MAP[resp]
                    break
                print("  Invalid choice. Please enter 1, 2, 3, 4, skip, or q.")

            if skip_item:
                break

        if skip_item:
            continue

        notes = input("\nReviewer Notes (optional): ").strip()

        # Compute overall winner
        slm_wins = sum(1 for v in field_judgments.values() if v == "slm_right")
        leg_wins = sum(1 for v in field_judgments.values() if v == "legacy_right")
        if slm_wins > leg_wins:
            overall = "slm_right"
        elif leg_wins > slm_wins:
            overall = "legacy_right"
        elif all(v == "both_right" for v in field_judgments.values()):
            overall = "both_right"
        else:
            overall = "neutral"

        label_record = {
            "fileHash": fhash,
            "labeledAt": datetime.now(timezone.utc).isoformat(),
            "overallJudgment": overall,
            "fieldJudgments": field_judgments,
            "notes": notes,
        }

        labels[fhash] = label_record
        labeled_count += 1
        update_supabase_shadow_label(fhash, label_record)

    existing_store["labels"] = labels
    save_review_labels(labels_file, existing_store)
    print(f"\nLabeling session complete. Total items labeled in this run: {labeled_count}.")


def apply_batch_labels(labels_to_apply: List[Dict[str, Any]], labels_file: Path) -> int:
    """Applies a list of labels in batch mode (for testing or automation)."""
    existing_store = load_review_labels(labels_file)
    labels = existing_store.get("labels", {})
    count = 0
    for item in labels_to_apply:
        fhash = item.get("fileHash")
        if not fhash:
            continue
        labels[fhash] = item
        update_supabase_shadow_label(fhash, item)
        count += 1
    existing_store["labels"] = labels
    save_review_labels(labels_file, existing_store)
    return count


def main():
    parser = argparse.ArgumentParser(description="Label ATS extraction disagreements.")
    parser.add_argument("--sample-file", type=str, help="Path to review sample JSON file.")
    parser.add_argument("--labels-file", type=str, default="data/review_labels.json", help="Path to save labels.")
    parser.add_argument("--batch-file", type=str, help="Apply batch labels from JSON file.")
    args = parser.parse_args()

    labels_path = PROJECT_ROOT / args.labels_file

    if args.batch_file and os.path.exists(args.batch_file):
        with open(args.batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)
        applied = apply_batch_labels(batch_data, labels_path)
        logger.info("Successfully applied %d batch labels to %s", applied, labels_path)
        return

    # If no sample file specified, search latest in data/review_samples
    sample_file_path = None
    if args.sample_file:
        sample_file_path = Path(args.sample_file)
    else:
        samples_dir = PROJECT_ROOT / "data" / "review_samples"
        if samples_dir.exists():
            files = sorted(samples_dir.glob("review_sample_*.json"), key=os.path.getmtime, reverse=True)
            if files:
                sample_file_path = files[0]

    if not sample_file_path or not sample_file_path.exists():
        logger.warning("No sample file found. Generating a review sample first...")
        from scripts.generate_review_sample import main as gen_sample
        gen_sample()
        files = sorted((PROJECT_ROOT / "data" / "review_samples").glob("review_sample_*.json"), key=os.path.getmtime, reverse=True)
        if files:
            sample_file_path = files[0]

    if sample_file_path and sample_file_path.exists():
        logger.info("Starting interactive labeling for %s", sample_file_path)
        run_interactive_labeling(sample_file_path, labels_path)
    else:
        logger.error("Could not find or generate review sample file.")


if __name__ == "__main__":
    main()
