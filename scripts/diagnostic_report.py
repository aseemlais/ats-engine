"""Run the SYNTHETIC diagnostic JD + resume (tests/fixtures) through the full pipeline and print a JSON report.

Usage (from the project root):   python scripts/diagnostic_report.py [--jd JD.txt --resume RESUME.txt]
It tries the real embedding model first. If the model cannot be loaded it re-runs in explicit DEGRADED mode
(semantic component excluded, flagged, NOT comparable with full-model scores) and says so in the output.
BM25 uses the persisted reference corpus in data/bm25_index.pkl if present (never modified by this script).
"""
import argparse, json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import config
from app.services.bm25_search import BM25SearchService
from app.services.certification_matcher import CertificationMatcherService
from app.services.embedding_service import EmbeddingService, EmbeddingUnavailableError
from app.services.hybrid_ranker import HybridRankerService
from app.services.jd_parser import JDParserService
from app.services.nlp_extractor import NLPExtractorService
from app.services.semantic_search import SemanticSearchService

FIX = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", default=str(FIX / "synthetic_trainee_jd.txt"))
    ap.add_argument("--resume", default=str(FIX / "synthetic_trainee_resume.txt"))
    a = ap.parse_args()
    jd_text, res_text = Path(a.jd).read_text(encoding="utf-8"), Path(a.resume).read_text(encoding="utf-8")
    jd = JDParserService().parse_job_description(jd_text, "DIAG")
    r = NLPExtractorService().extract_structured("DIAG_CAND", "diag", "h", res_text, res_text)
    emb = EmbeddingService()
    bm = BM25SearchService()                                   # reads the persisted corpus; build_index(persist=False) below
    rk = HybridRankerService(semantic_service=SemanticSearchService(embedding_service=emb, qdrant_db=object()),
                             bm25_service=bm, certification_service=CertificationMatcherService(emb))
    note = "full model"
    try:
        vecs = {r.candidate_id: emb.compute_resume_vectors(r)}
    except EmbeddingUnavailableError as e:
        note = f"DEGRADED (semantic excluded): {e}"
        config.allow_degraded_semantic = True
        vecs = {}
    resp = rk.rank_candidates([r], vecs, jd, top_k=1)
    b = resp.results[0]
    print(json.dumps({
        "scoring_mode_note": note,
        "jd": {"title": jd.title, "required": jd.required_skills, "preferred": jd.preferred_skills,
               "supporting": jd.supporting_skills, "nonmandatory": jd.nonmandatory_skills,
               "accepted_alternatives": jd.skill_alternatives, "minimum_experience_years": jd.minimum_experience_years,
               "experience_applicable": jd.has_experience_requirement, "education": jd.education_requirements,
               "certification_requirement": rk.certification_service.requirement_level(jd)},
        "candidate": {"skills": sorted(r.skills), "work_years": r.work_experience_years, "internship_years": r.internship_years},
        "result": {"final_score": b.final_score, "eligibility": b.eligibility, "mandatory_requirements_met": b.mandatory_requirements_met,
                   "missing_mandatory": b.missing_mandatory_skills, "matched_required": b.matched_required_skills,
                   "accepted_alternatives_used": b.accepted_alternatives, "degraded": b.degraded, "components": b.components,
                   "bm25_reference_docs": len(bm.reference_corpus)},
    }, indent=2))


if __name__ == "__main__":
    main()
