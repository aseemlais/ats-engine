from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class CandidateScoreBreakdown(BaseModel):
    candidate_id: str
    name: Optional[str] = "Unknown"
    rank: int = 1
    final_score: float = 0.0  # Percentage 0.0 to 100.0
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    skill_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    certification_score: float = 0.0
    matched_required_skills: List[str] = Field(default_factory=list)
    missing_required_skills: List[str] = Field(default_factory=list)
    matched_preferred_skills: List[str] = Field(default_factory=list)
    candidate_experience_years: float = 0.0
    required_experience_years: float = 0.0
    education_tier: str = "none"
    explanation: str = ""
    bm25_diagnostics: Dict[str, Any] = Field(default_factory=dict)   # query, raw/normalised score, reference size, matched terms
    # Added (optional, backward compatible)
    certification_bonus: float = 0.0            # points (0-100 scale) added for relevant certifications when the JD is silent on them
    raw_ats_score: Optional[float] = None       # = final_score: the unmodified formula output
    percentile_rank: Optional[float] = None     # relative to this request's candidates only
    eligibility: str = "Eligible"
    fit: Optional[str] = None
    recommendation: Optional[str] = None
    certification_level: str = "none"           # none | preferred | required (from the JD)
    certification_effective_score: Optional[float] = None  # value actually used in the weighted sum
    skill_applicable: bool = True
    # --- explainability (all optional / backward compatible) ---
    # Per-component {score, status, nominal_weight, effective_weight, contribution}; contributions sum to final_score/100.
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    experience_applicable: bool = True          # False = JD states no minimum experience (component excluded, not scored 0)
    scoring_mode: str = "full"                  # "full" | "degraded_no_semantic" (NOT comparable with full-mode scores)
    degraded: bool = False
    mandatory_requirements_met: bool = True     # independent of fit score and of the (lenient/strict) filter mode
    missing_mandatory_skills: List[str] = Field(default_factory=list)
    eligibility_reasons: List[str] = Field(default_factory=list)
    accepted_alternatives: List[str] = Field(default_factory=list)   # e.g. "mysql / sql (any one): satisfied by sql"
    matched_supporting_skills: List[str] = Field(default_factory=list)
    missing_preferred_skills: List[str] = Field(default_factory=list)

class SearchResponse(BaseModel):
    job_id: str
    total_indexed_resumes: int
    hard_filtered_count: int
    retrieved_count: int
    final_count: int
    execution_time_seconds: float
    weights_used: Dict[str, float]
    degraded_count: int = 0                     # results scored without the semantic component
    results: List[CandidateScoreBreakdown]
