from typing import List, Optional
from app.models.candidate import CandidateScoreBreakdown
from app.models.job import JobDescription

class ExplainabilityService:
    def generate_explanation(self, score_breakdown: CandidateScoreBreakdown, jd: JobDescription) -> str:
        """
        Generates a clean, transparent, deterministic explanation without LLM hallucination.
        """
        sb = score_breakdown
        names = {"semantic": "Semantic Similarity", "bm25": "BM25 Lexical Match", "skill": "Skill Match",
                 "experience": "Experience Match", "certification": "Certification Match"}
        lines = [
            f"Candidate: {sb.candidate_id} ({sb.name or 'Unknown'})",
            f"Final ATS Score: {sb.final_score:.1f}% (Rank #{sb.rank})",
            f"Mandatory requirements: {'met' if sb.mandatory_requirements_met else 'NOT met'}"
            + (f" - missing: {', '.join(sb.missing_mandatory_skills)}" if sb.missing_mandatory_skills else ""),
            f"Eligibility: {sb.eligibility}" + (f" ({'; '.join(sb.eligibility_reasons)})" if sb.eligibility_reasons else ""),
        ]
        if sb.degraded:
            lines.append("WARNING: degraded scoring mode - semantic similarity unavailable; score is NOT comparable with full-mode scores.")
        lines += ["", "Component Breakdown (score | weight actually applied | contribution to final):"]
        if sb.components:
            for key, label in names.items():
                c = sb.components.get(key)
                if not c:
                    continue
                if c["status"] == "applied":
                    lines.append(f"  - {label}: {c['score'] * 100:.1f}% | weight {c['effective_weight'] * 100:.1f}% | +{c['contribution'] * 100:.1f} pts")
                else:
                    why = {"not_applicable": "not applicable to this JD", "neutral": "optional, not held - no penalty",
                           "unavailable": "unavailable"}.get(c["status"], c["status"])
                    lines.append(f"  - {label}: excluded ({why})")
        else:
            lines += [f"  - Semantic Similarity:   {sb.semantic_score * 100:.1f}%", f"  - BM25 Lexical Match:    {sb.bm25_score * 100:.1f}%",
                      f"  - Skill Match:           {sb.skill_score * 100:.1f}%", f"  - Experience Match:      {sb.experience_score * 100:.1f}%",
                      f"  - Certification Match:   {sb.certification_score * 100:.1f}%"]
        lines += [f"  - Education Match (informational): {sb.education_score * 100:.1f}%", "", "Skills Analysis:"]
        for alt in sb.accepted_alternatives:
            lines.append(f"  Accepted alternative: {alt}")

        if score_breakdown.matched_required_skills:
            lines.append("  Matched Required Skills:")
            for s in score_breakdown.matched_required_skills:
                lines.append(f"    ✓ {s}")
        else:
            lines.append("  Matched Required Skills: None")

        if score_breakdown.missing_required_skills:
            lines.append("  Missing Required Skills:")
            for s in score_breakdown.missing_required_skills:
                lines.append(f"    ✗ {s}")
        else:
            lines.append("  Missing Required Skills: None (100% matched)")

        if score_breakdown.matched_preferred_skills:
            lines.append(f"  Matched Preferred Skills: {', '.join(score_breakdown.matched_preferred_skills)}")

        lines.append("")
        lines.append("Experience Analysis:")
        if not score_breakdown.experience_applicable:
            lines.append("  No minimum experience required by the JD (component not scored).")
        else:
            lines.append(f"  Required:  {score_breakdown.required_experience_years:.1f}+ years")
        lines.append(f"  Candidate: {score_breakdown.candidate_experience_years:.1f} years")

        lines.append("")
        lines.append(f"Education Tier: {score_breakdown.education_tier.title()}")

        return "\n".join(lines)
