from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from app.config import config
from app.services.embedding_service import EmbeddingUnavailableError
from app.models.job import JobDescription
from app.models.candidate import SearchResponse
from app.services.jd_parser import JDParserService
from app.services.hybrid_ranker import HybridRankerService
from app.services.embedding_service import EmbeddingService
from app.database.storage import MetadataStorage

router = APIRouter(prefix="/search", tags=["Search"])

jd_parser = JDParserService()
ranker = HybridRankerService()
embedding_service = EmbeddingService()
storage = MetadataStorage()

# In-memory vectors cache for fast repeated queries
_VECTORS_CACHE: Dict[str, Dict[str, Any]] = {}

def get_vectors_cache():
    global _VECTORS_CACHE
    if not _VECTORS_CACHE:
        resumes = storage.get_all_resumes()
        for r in resumes:
            _VECTORS_CACHE[r.candidate_id] = embedding_service.compute_resume_vectors(r)
    return _VECTORS_CACHE

class SearchRequest(BaseModel):
    job_description: str
    job_id: Optional[str] = "JD_SEARCH"
    top_k: int = Field(default=100, ge=1, le=500)
    semantic_weight: Optional[float] = None
    bm25_weight: Optional[float] = None
    skill_weight: Optional[float] = None
    experience_weight: Optional[float] = None

@router.post("", response_model=SearchResponse)
def search_resumes(req: SearchRequest):
    """
    Executes hybrid semantic search + BM25 + skill/experience matching on indexed resumes.
    Returns ranked shortlists with deterministic score explanations.
    """
    all_resumes = storage.get_all_resumes()
    if not all_resumes:
        raise HTTPException(status_code=400, detail="No resumes have been indexed yet. Please upload resumes first.")

    # Parse JD
    parsed_jd = jd_parser.parse_job_description(req.job_description, job_id=req.job_id or "JD_SEARCH")

    # Per-request overrides take precedence over environment/.env/default weights (certification is not overridable here)
    overrides = {"semantic": req.semantic_weight, "bm25": req.bm25_weight,
                 "skill": req.skill_weight, "experience": req.experience_weight}
    custom_weights = config.scoring_weights(overrides) if any(v is not None for v in overrides.values()) else None

    vec_cache = get_vectors_cache()

    try:
        response = ranker.rank_candidates(
            resumes=all_resumes,
            resume_vectors_map=vec_cache,
            jd=parsed_jd,
            top_k=req.top_k,
            weights=custom_weights
        )
    except EmbeddingUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return response
