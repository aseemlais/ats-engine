"""
Score diagnostics: per-resume component table + distributions for one job description.
Usage (from project root):  python scripts/score_diagnostics.py --jd data/resumes/JD_Senior_Data_Analyst.txt --out /tmp/diag.json
Uses the REAL embedding model when available; otherwise semantic/cert-semantic are reported as NaN
(nothing is imputed). Reads resumes from data/ats_metadata.db (parsed_json) - read-only.
"""
import argparse, json, math, sqlite3, sys, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

try:
    import sentence_transformers  # noqa
    HAVE_MODEL = True
except Exception:
    st = types.ModuleType("sentence_transformers"); st.SentenceTransformer = object
    sys.modules["sentence_transformers"] = st
    HAVE_MODEL = False

from app.config import config
from app.models.resume import ParsedResume
from app.services.jd_parser import JDParserService
from app.services.bm25_search import BM25SearchService
from app.services.skill_matcher import SkillMatcherService
from app.services.experience_matcher import ExperienceMatcherService
from app.services.hard_filter import HardFilterService
from app.services.nlp_extractor import NLPExtractorService


def load_resumes(db, reparse):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for cid, pj in con.execute("select candidate_id, parsed_json from resumes order by candidate_id"):
        r = ParsedResume(**json.loads(pj))
        if reparse:  # re-run the CURRENT extractor on stored text so extractor fixes are visible
            r = NLPExtractorService().extract_structured(r.candidate_id, r.source_file, r.file_hash, r.raw_text, r.cleaned_text)
        out.append(r)
    return out


def stats(a):
    a = np.array([x for x in a if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(a) == 0:
        return {}
    q = np.percentile(a, [10, 25, 50, 75, 90])
    return dict(n=len(a), min=a.min(), p10=q[0], p25=q[1], median=q[2], mean=a.mean(), p75=q[3], p90=q[4], max=a.max())


def semantic_report(resumes, jd, emb):
    """Raw weighted cosine vs calibrated (x-0.15)/0.55: shows how much of the pool is clipped at 0 or 1."""
    from app.services.semantic_search import SemanticSearchService
    vecs = emb.compute_corpus_vectors_batched(resumes)
    jv = emb.compute_jd_vectors(jd)
    svc = SemanticSearchService(embedding_service=emb, qdrant_db=object())
    raws, cals = [], []
    for r in resumes:
        cal = svc.compute_section_similarity(vecs[r.candidate_id], jv)
        raws.append(cal)  # placeholder, replaced below
        cals.append(cal)
    # recompute raw (pre-calibration) from the same pieces
    raws = []
    for r in resumes:
        v = vecs[r.candidate_id]
        tot = 0.0
        for sec, wt in (("skills", config.section_skills_weight), ("experience", config.section_experience_weight),
                        ("projects", config.section_projects_weight), ("education", config.section_education_weight)):
            tot += wt * svc.compute_cosine_similarity(v[sec], jv[sec])
        raws.append(tot)
    raws = np.array(raws)
    print("semantic RAW weighted cosine :", {k: round(float(v), 3) for k, v in stats(list(raws)).items()})
    print("semantic CALIBRATED          :", {k: round(float(v), 3) for k, v in stats(cals).items()})
    print(f"clipped at 0 (raw<=0.15): {np.mean(raws <= 0.15):.0%}   clipped at 1 (raw>=0.70): {np.mean(raws >= 0.70):.0%}")
    print("model max_seq_length:", getattr(getattr(emb, "model", None), "max_seq_length", "n/a"),
          "| resumes longer than ~190 words:", f"{np.mean([len(r.cleaned_text.split()) > 190 for r in resumes]):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True); ap.add_argument("--db", default=str(config.data_dir / "ats_metadata.db"))
    ap.add_argument("--out"); ap.add_argument("--reparse", action="store_true")
    a = ap.parse_args()
    jd = JDParserService().parse_job_description(Path(a.jd).read_text(), job_id="JD")
    resumes = load_resumes(a.db, a.reparse)
    # BM25: temp service so the persisted index is never touched
    tmp_idx = Path("/tmp/_diag_bm25.pkl")
    tmp_idx.unlink(missing_ok=True)
    bm = BM25SearchService(index_path=tmp_idx)
    kw = {"persist": False} if "persist" in BM25SearchService.build_index.__code__.co_varnames else {}
    # the full pool doubles as the stored reference corpus (written to a TEMP file, never data/bm25_index.pkl)
    bm.build_index([r.candidate_id for r in resumes], [r.cleaned_text for r in resumes])
    batch = bm.search(jd.raw_text, top_k=len(resumes))
    single = {}
    for r in resumes:
        bm.build_index([r.candidate_id], [r.cleaned_text], **kw)
        single[r.candidate_id] = bm.search(jd.raw_text).get(r.candidate_id, 0.0)
    hf = HardFilterService(enabled=True, strict=False)
    rows = []
    for r in resumes:
        sk = SkillMatcherService().evaluate_skills(r, jd)
        ex = ExperienceMatcherService().evaluate_experience(r, jd)
        ok, why = hf.passes_filter(r, jd)
        rows.append(dict(id=r.candidate_id, name=r.name, bm25_batch=batch[r.candidate_id], bm25_single=single[r.candidate_id],
                         skill=sk[0], matched=sk[1], missing=sk[2], exp=ex, exp_years=r.experience_years,
                         work_years=r.work_experience_years, intern_years=r.internship_years,
                         n_skills=len(r.skills), has_certs=bool(r.certifications), eligible=ok, why=why))
    print(f"model_available={HAVE_MODEL} resumes={len(resumes)} JD title={jd.title!r} min_exp={jd.minimum_experience_years}")
    print("required:", jd.required_skills, "| preferred:", jd.preferred_skills, "| mandatory:", jd.mandatory_skills)
    for k in ("bm25_batch", "bm25_single", "skill", "exp"):
        print(k, {m: round(float(v), 3) for m, v in stats([x[k] for x in rows]).items()})
    print("eligible:", sum(x["eligible"] for x in rows), "/", len(rows))
    if HAVE_MODEL:
        from app.services.embedding_service import EmbeddingService
        semantic_report(resumes, jd, EmbeddingService())
    else:
        print("semantic: skipped (sentence-transformers / model not available here)")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=str))

if __name__ == "__main__":
    main()
