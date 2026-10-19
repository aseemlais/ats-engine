"""Transparent composition of the final fit score from component scores (pure Python, no model needed).

Component status
  applied         scored and included in the weighted sum
  not_applicable  the JD does not ask for it (no experience minimum, no certification mention, no recognisable
                  skills) -> excluded, never rewarded or penalised
  neutral         a PREFERRED certification the candidate does not hold -> excluded (absence is not penalised)
  unavailable     cannot be computed (semantic model unavailable / candidate has no vectors) -> excluded and the
                  result is flagged degraded

Formula (identical for every candidate; weights are configurable and need not sum to 1):
    active  = components with status "applied"
    final   = sum(w_i * s_i for i in active) / sum(w_i for i in active)            in [0, 1]
    effective weight_i = w_i / sum(w_j for j in active)   ->  contributions c_i = effective_i * s_i  sum to `final`
This is mathematically the same as "a missing component takes the candidate's own weighted average of the other
components" (the earlier neutral-imputation policy), so a candidate is never rewarded or penalised for a
criterion that does not apply. Certification influence is capped by its weight: a held preferred certificate only
counts when its score exceeds the candidate's average on the other components, and can add at most that weight.
"""
from typing import Dict, Optional

COMPONENTS = ("semantic", "bm25", "skill", "experience", "certification")


def compose_score(scores: Dict[str, Optional[float]], status: Dict[str, str], weights: Dict[str, float]) -> Dict:
    """scores: raw 0-1 score per component (None allowed when not applied). status: component -> status string.
    Returns {"final": 0-1, "components": {name: {score,status,nominal_weight,effective_weight,contribution}}}."""
    nominal = {c: max(float(weights.get(c, 0.0)), 0.0) for c in COMPONENTS}
    active = [c for c in COMPONENTS if status.get(c) == "applied" and nominal[c] > 0 and scores.get(c) is not None]
    total_w = sum(nominal[c] for c in active)
    comps: Dict[str, Dict] = {}
    final = 0.0
    for c in COMPONENTS:
        s = scores.get(c)
        s = None if s is None else min(max(float(s), 0.0), 1.0)
        if c in active:
            eff = nominal[c] / total_w
            contrib = eff * s
            final += contrib
            comps[c] = {"score": round(s, 4), "status": "applied", "nominal_weight": round(nominal[c], 4),
                        "effective_weight": round(eff, 4), "contribution": round(contrib, 4)}
        else:
            st = status.get(c, "not_applicable") if status.get(c) != "applied" else "not_applicable"  # applied with 0 weight
            # no score is reported for components that were not computed for this JD/candidate (no fabricated 0.0);
            # a 'neutral' (optional, not held) component keeps the score it was evaluated at
            shown = None if (s is None or st in ("not_applicable", "unavailable")) else round(s, 4)
            comps[c] = {"score": shown, "status": st,
                        "nominal_weight": round(nominal[c], 4), "effective_weight": 0.0, "contribution": 0.0}
    return {"final": min(max(final, 0.0), 1.0), "components": comps}


def certification_status(level: str, cert_score: float, other_avg: float, legacy: bool = False) -> str:
    """'required' -> applied (absence scores 0.0, a genuine miss); 'preferred' -> applied only when the held
    certificate's score beats the candidate's average on the other components (capped benefit), else 'neutral';
    'none' -> not_applicable. legacy keeps the old always-applied behaviour."""
    if legacy or level == "required":
        return "applied"
    if level == "preferred":
        return "applied" if cert_score > other_avg else "neutral"
    return "not_applicable"
