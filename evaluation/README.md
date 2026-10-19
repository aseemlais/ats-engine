# Labelled JD-resume evaluation (support only - NO labelled data exists yet)

No evaluation result has been produced or is claimed. This folder only prepares the format.

1. Copy `labelled_pairs_TEMPLATE.csv` to `labelled_pairs.csv` and fill one row per human-judged pair:
   `jd_file` / `resume_file` (paths relative to the project root), `human_label` (integer 0 = not relevant, 1 = partial,
   2 = relevant, 3 = strong match), `labeller`, `notes`. Use several labellers per pair where possible.
2. Run: `python scripts/evaluate_bm25_labelled.py evaluation/labelled_pairs.csv`
   It scores every pair with each BM25 normalisation method and reports Spearman rank correlation and, per JD,
   nDCG@k against the human labels. Fewer than 30 labelled pairs is reported as "too few to conclude".
3. Hold out some JDs before tuning anything; never tune weights on the pairs you report.
