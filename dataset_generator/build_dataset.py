import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

# Ensure parent directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset_generator.augmenter import augment_resume_text
from dataset_generator.document_renderer import render_and_extract_text
from dataset_generator.profile_generator import generate_candidate_profile
from app.services.prompt_templates import EXTRACTION_SYSTEM_PROMPT

SYSTEM_PROMPT = EXTRACTION_SYSTEM_PROMPT

PROFILE_TYPES = ["standard", "parenthesized_title", "fresher", "overlapping", "career_gap", "missing_contact"]


def create_chat_example(raw_text: str, ground_truth_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats the example into the standard OpenAI / Hugging Face chat format
    compatible with Qwen2.5-1.5B-Instruct SFT (Unsloth, TRL, SFTTrainer).
    """
    clean_gt = {
        "candidateName": ground_truth_profile.get("candidateName"),
        "email": ground_truth_profile.get("email"),
        "phone": ground_truth_profile.get("phone"),
        "location": ground_truth_profile.get("location"),
        "jobTitle": ground_truth_profile.get("jobTitle"),
        "skills": ground_truth_profile.get("skills", []),
        "education": ground_truth_profile.get("education", []),
        "workExperience": ground_truth_profile.get("workExperience", []),
        "certifications": ground_truth_profile.get("certifications", []),
        "projects": ground_truth_profile.get("projects", []),
    }

    assistant_json = json.dumps(clean_gt, ensure_ascii=False, separators=(",", ":"))

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"RESUME TEXT:\n{raw_text}"},
            {"role": "assistant", "content": assistant_json},
        ]
    }


def generate_dataset_split(
    num_samples: int,
    templates: List[str],
    split_name: str,
    output_path: str,
    augment: bool = True,
) -> int:
    """
    Generates and saves a JSONL split strictly using the provided template types.
    """
    records = []
    print(f"Generating {num_samples} samples for '{split_name}' using templates: {templates}...")

    for i in range(num_samples):
        # Round-robin profile types to ensure balanced edge-case representation
        p_type = PROFILE_TYPES[i % len(PROFILE_TYPES)]
        profile = generate_candidate_profile(profile_type=p_type)

        # Select template from the split's allowed templates
        template = random.choice(templates)
        raw_text, fmt = render_and_extract_text(profile, template=template)

        # Apply realistic noise augmentation
        if augment:
            noisy_text = augment_resume_text(raw_text, candidate_name=profile.get("candidateName", ""))
        else:
            noisy_text = raw_text

        chat_example = create_chat_example(noisy_text, profile)
        records.append(chat_example)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} samples to {output_path}")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SFT dataset for Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-size", type=int, default=200, help="Number of training samples")
    parser.add_argument("--val-size", type=int, default=40, help="Number of validation samples")
    parser.add_argument("--test-size", type=int, default=40, help="Number of test samples")
    parser.add_argument("--gold-size", type=int, default=20, help="Number of held-out gold evaluation samples")
    parser.add_argument("--output-dir", type=str, default="data/sft", help="Output directory for JSONL files")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Strict template split to avoid layout data leakage:
    # Train: single_column_pdf and docx
    # Val: docx and two_column_pdf
    # Test: two_column_pdf (unseen two-column layout for single-column trained models)
    # Gold: mixed diverse layouts for human/ground-truth benchmarking
    train_templates = ["single_column_pdf", "docx"]
    val_templates = ["docx", "two_column_pdf"]
    test_templates = ["two_column_pdf"]
    gold_templates = ["single_column_pdf", "two_column_pdf", "docx"]

    print("=== Phase 2: Starting Synthetic Dataset Generation ===")
    n_train = generate_dataset_split(
        args.train_size, train_templates, "train", os.path.join(out_dir, "train.jsonl"), augment=True
    )
    n_val = generate_dataset_split(
        args.val_size, val_templates, "val", os.path.join(out_dir, "val.jsonl"), augment=True
    )
    n_test = generate_dataset_split(
        args.test_size, test_templates, "test", os.path.join(out_dir, "test.jsonl"), augment=True
    )
    n_gold = generate_dataset_split(
        args.gold_size, gold_templates, "gold_eval", os.path.join(out_dir, "gold_eval.jsonl"), augment=False
    )

    stats = {
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        "gold_eval_samples": n_gold,
        "total_generated": n_train + n_val + n_test + n_gold,
        "train_templates": train_templates,
        "val_templates": val_templates,
        "test_templates": test_templates,
        "gold_templates": gold_templates,
        "edge_case_profiles": PROFILE_TYPES,
        "output_directory": out_dir,
    }

    stats_file = os.path.join(out_dir, "dataset_summary.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Dataset Generation Complete ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
