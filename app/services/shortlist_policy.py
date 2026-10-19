"""Shortlist policy, kept separate from scoring. Nothing here changes a numeric score."""
from typing import List, Optional, Sequence, Tuple

from app.config import config


def classify_fit(score: float, shortlist_thr: Optional[float] = None, review_thr: Optional[float] = None) -> str:
    st = config.shortlist_threshold if shortlist_thr is None else shortlist_thr
    rt = config.review_threshold if review_thr is None else review_thr
    return "Fit" if score >= st else ("Review" if score >= rt else "Not Fit")


def recommend_single(score: float, eligible: bool, shortlist_thr: Optional[float] = None,
                     review_thr: Optional[float] = None) -> Tuple[str, str]:
    """(fit, recommendation). Ineligible candidates are never Shortlisted/Review: a mandatory requirement
    is not silently waived. `score` should be the unrounded ATS score."""
    fit = classify_fit(score, shortlist_thr, review_thr)
    if not eligible:
        return fit, "Not Shortlisted"
    return fit, {"Fit": "Shortlisted", "Review": "Review"}.get(fit, "Not Shortlisted")


def percentile_ranks(scores: Sequence[float]) -> List[float]:
    """Mid-rank percentile (0-100) of each score within this list (ties share the mean). Relative only."""
    n = len(scores)
    return [round(100.0 * (sum(x < s for x in scores) + 0.5 * sum(x == s for x in scores)) / n, 1) for s in scores]


def apply_batch_policy(candidates: list, k: Optional[int] = None) -> None:
    """
    In place on CandidateScoreBreakdown objects sorted by score (desc), all hard-filter eligible.
    >= shortlist threshold -> Shortlisted; review band -> Review; else Not Shortlisted. If nobody reaches
    the shortlist threshold, the top-K eligible by actual score are marked 'Best Available Match' (human
    review); with fewer than K candidates all are returned.
    """
    k = config.best_available_k if k is None else k
    for c in candidates:   # decisions use the UNROUNDED score when the ranker supplied it
        if c.raw_ats_score is None:
            c.raw_ats_score = c.final_score
    pcts = percentile_ranks([c.raw_ats_score for c in candidates])
    for c, pct in zip(candidates, pcts):
        c.percentile_rank = pct
        c.fit, c.recommendation = recommend_single(c.raw_ats_score, c.eligibility == "Eligible")
    if candidates and not any(c.recommendation == "Shortlisted" for c in candidates):
        for c in candidates[:k]:
            if c.eligibility == "Eligible" and c.recommendation != "Review":
                c.recommendation = "Best Available Match"
