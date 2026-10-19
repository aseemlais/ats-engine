import time
from typing import List, Dict, Optional, Tuple
import numpy as np

from app.config import config
from app.models.resume import ParsedResume
from app.models.job import JobDescription
from app.models.candidate import CandidateScoreBreakdown, SearchResponse
from app.services.semantic_search import SemanticSearchService
from app.services.bm25_search import BM25SearchService
from app.services.bm25_query import build_bm25_query
from app.services.skill_matcher import SkillMatcherService
from app.services.experience_matcher import ExperienceMatcherService
from app.services.education_matcher import EducationMatcherService
from app.services.certification_matcher import CertificationMatcherService
from app.services.hard_filter import HardFilterService
from app.services.explainability import ExplainabilityService
from app.services.shortlist_policy import apply_batch_policy
from app.services.embedding_service import EmbeddingUnavailableError
from app.services.score_composer import certification_status, compose_score

class HybridRankerService:
    def __init__(
        self,
        semantic_service: Optional[SemanticSearchService] = None,
        bm25_service: Optional[BM25SearchService] = None,
        skill_service: Optional[SkillMatcherService] = None,
        experience_service: Optional[ExperienceMatcherService] = None,
        education_service: Optional[EducationMatcherService] = None,
        certification_service: Optional[CertificationMatcherService] = None,
        hard_filter_service: Optional[HardFilterService] = None,
        explainability_service: Optional[ExplainabilityService] = None
    ):
        self.semantic_service = semantic_service or SemanticSearchService()
        self.bm25_service = bm25_service or BM25SearchService()
        self.skill_service = skill_service or SkillMatcherService()
        self.experience_service = experience_service or ExperienceMatcherService()
        self.education_service = education_service or EducationMatcherService()
        self.certification_service = certification_service or CertificationMatcherService(self.semantic_service.embedding_service)
        self.hard_filter_service = hard_filter_service or HardFilterService()
        self.explainability_service = explainability_service or ExplainabilityService()
        self.shortlist_threshold = config.shortlist_threshold
        self.review_threshold = config.review_threshold

    def _bm25_diagnostics(self, q, res, cid: str) -> Dict:
        """Explainability for the lexical component. raw_score is NOT a probability or a percentage of skills covered."""
        info = self.bm25_service.reference_info()
        return {
            "query_terms": res.unique_terms, "query_method": q.method, "fallback_reason": q.fallback_reason,
            "query_mode": q.mode, "generic_terms_removed": q.removed_terms,
            "raw_score": round(res.raw.get(cid, 0.0), 4), "normalized_score": round(res.normalized.get(cid, 0.0), 4),
            "normalization_method": res.method, "normalization_denominator": round(res.denominator, 4),
            "percentile_vs_reference": round(res.percentile.get(cid, 0.0), 4),
            "reference_documents": res.reference_size, "reference_reliability": res.reference_reliability,
            "matched_terms": res.matched_terms.get(cid, []), "unmatched_terms": res.unmatched_terms.get(cid, []),
            "unseen_terms": res.unseen_terms, "index_stale": info["stale"], "notes": res.notes,
            "semantics": "normalized = raw / theoretical maximum over seen query terms (method-dependent); not a probability",
        }

    def rank_candidates(
        self,
        resumes: List[ParsedResume],
        resume_vectors_map: Dict[str, Dict[str, np.ndarray]],
        jd: JobDescription,
        top_k: int = 100,
        weights: Optional[Dict[str, float]] = None
    ) -> SearchResponse:
        """
        Executes complete end-to-end multi-stage pipeline:
        Resumes -> Hard Filtering -> Hybrid Retrieval -> Deterministic Scoring -> Top-K Ranking
        """
        t0 = time.time()
        total_resumes = len(resumes)

        # Configurable weights: per-request overrides > environment/.env > defaults (see app/config.py)
        w = config.scoring_weights(weights)

        # 1. HARD FILTERING STAGE
        passed_resumes, rejections = self.hard_filter_service.filter_candidates(resumes, jd)
        hard_filtered_count = len(passed_resumes)

        if not passed_resumes:
            return SearchResponse(
                job_id=jd.job_id,
                total_indexed_resumes=total_resumes,
                hard_filtered_count=0,
                retrieved_count=0,
                final_count=0,
                execution_time_seconds=time.time() - t0,
                weights_used=w,
                results=[]
            )

        # 2. LEXICAL BM25 RETRIEVAL
        # Stateless scoring against the persisted reference corpus: nothing shared is mutated per request, so
        # concurrent requests cannot interfere and the stored index is never overwritten.
        self.bm25_service.refresh_if_stale()          # explicit reload if ingestion rewrote the index file
        bm25_query = build_bm25_query(jd)             # parsed technical skills + responsibilities, administrative text excluded
        bm25_result = self.bm25_service.score_documents(
            bm25_query.text, [r.candidate_id for r in passed_resumes], [r.cleaned_text for r in passed_resumes])
        bm25_scores = bm25_result.normalized

        # 3. SEMANTIC VECTOR REPRESENTATIONS
        # If the embedding model cannot be loaded the request fails loudly (EmbeddingUnavailableError), unless
        # ALLOW_DEGRADED_SEMANTIC=true, in which case the semantic component is excluded and results are flagged.
        jd_vectors = None
        try:
            jd_vectors = self.semantic_service.embedding_service.compute_jd_vectors(jd)
        except EmbeddingUnavailableError:
            if not config.allow_degraded_semantic:
                raise

        # JD-level applicability of optional criteria (same for every candidate in the request)
        cert_level = self.certification_service.requirement_level(jd)
        skill_applicable = self.skill_service.is_applicable(jd)
        exp_applicable = self.experience_service.is_applicable(jd)
        legacy = config.na_policy == "legacy"

        # 4. HYBRID SCORING FOR CANDIDATES (see app/services/score_composer.py for the exact formula)
        scored_candidates: List[CandidateScoreBreakdown] = []

        for resume in passed_resumes:
            cid = resume.candidate_id

            cand_vectors = resume_vectors_map.get(cid)
            sem_ok = jd_vectors is not None and bool(cand_vectors)
            semantic_score = self.semantic_service.compute_section_similarity(cand_vectors, jd_vectors) if sem_ok else 0.0
            bm25_ok = cid in bm25_scores
            bm25_score = bm25_scores.get(cid, 0.0)

            sk = self.skill_service.evaluate_detailed(resume, jd)
            skill_score = sk["score"]
            exp_score = self.experience_service.evaluate_experience(resume, jd) if exp_applicable else 0.0
            cert_score = self.certification_service.evaluate_certifications(resume, jd)
            edu_score, highest_tier = self.education_service.evaluate_education(resume, jd)
            elig = self.hard_filter_service.assess(resume, jd)

            scores = {"semantic": semantic_score, "bm25": bm25_score, "skill": skill_score,
                      "experience": exp_score, "certification": cert_score}
            status = {
                "semantic": "applied" if sem_ok else "unavailable",
                "bm25": "applied" if bm25_ok else "not_applicable",
                "skill": "applied" if (skill_applicable or legacy) else "not_applicable",
                "experience": "applied" if exp_applicable else "not_applicable",
            }
            # certification: needs the candidate's average over the other applicable components
            core = compose_score(scores, {**status, "certification": "not_applicable"}, w)
            other_avg = core["final"]
            status["certification"] = certification_status(cert_level, cert_score, other_avg, legacy)
            composed = compose_score(scores, status, w)
            # JD silent on certifications: relevant ones add a small bounded bonus (0 when none is relevant); never a penalty
            cert_bonus = config.cert_bonus_none * cert_score * 100.0 if (cert_level == "none" and not legacy) else 0.0
            final_percentage = min(max(composed["final"] * 100.0 + cert_bonus, 0.0), 100.0)   # unrounded: used for ALL decisions
            degraded = not sem_ok

            breakdown = CandidateScoreBreakdown(
                candidate_id=cid,
                name=resume.name,
                rank=0,
                final_score=round(final_percentage, 2),
                raw_ats_score=final_percentage,
                certification_bonus=round(cert_bonus, 2),
                semantic_score=round(semantic_score, 4),
                bm25_score=round(bm25_score, 4),
                bm25_diagnostics=self._bm25_diagnostics(bm25_query, bm25_result, cid),
                skill_score=round(skill_score, 4),
                experience_score=round(exp_score, 4),
                certification_score=round(cert_score, 4),
                education_score=round(edu_score, 4),
                matched_required_skills=sk["matched_required"],
                missing_required_skills=sk["missing_required"],
                matched_preferred_skills=sk["matched_preferred"] + sk["matched_supporting"],
                candidate_experience_years=resume.experience_years,
                required_experience_years=jd.minimum_experience_years,
                education_tier=highest_tier,
                certification_level=cert_level,
                certification_effective_score=round(composed["components"]["certification"]["contribution"]
                                                    / max(composed["components"]["certification"]["effective_weight"], 1e-12), 4)
                if composed["components"]["certification"]["status"] == "applied" else round(other_avg, 4),
                skill_applicable=skill_applicable,
                components=composed["components"],
                experience_applicable=exp_applicable,
                scoring_mode="degraded_no_semantic" if degraded else "full",
                degraded=degraded,
                eligibility="Eligible" if elig["eligible"] else "Not Eligible",
                eligibility_reasons=elig["reasons"],
                mandatory_requirements_met=elig.get("mandatory_requirements_met", elig["eligible"]),
                missing_mandatory_skills=elig.get("missing_mandatory", []),
                accepted_alternatives=sk["alternatives_satisfied"],
                matched_supporting_skills=sk["matched_supporting"],
                missing_preferred_skills=sk["missing_preferred"],
            )
            scored_candidates.append(breakdown)

        # 5. SORTING & TOP-K SHORTLISTING
        scored_candidates.sort(key=lambda c: c.raw_ats_score if c.raw_ats_score is not None else c.final_score, reverse=True)

        apply_batch_policy(scored_candidates)  # percentile + recommendation; scores untouched
        top_candidates = scored_candidates[:top_k]
        for rank_idx, cand in enumerate(top_candidates, start=1):
            cand.rank = rank_idx
            cand.explanation = self.explainability_service.generate_explanation(cand, jd)

        elapsed = time.time() - t0

        return SearchResponse(
            job_id=jd.job_id,
            total_indexed_resumes=total_resumes,
            hard_filtered_count=hard_filtered_count,
            retrieved_count=len(scored_candidates),
            final_count=len(top_candidates),
            execution_time_seconds=round(elapsed, 4),
            weights_used=w,
            degraded_count=sum(1 for c in scored_candidates if c.degraded),
            results=top_candidates
        )
