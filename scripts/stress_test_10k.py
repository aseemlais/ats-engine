import time
import tracemalloc
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config
from app.models.resume import ParsedResume
from app.models.job import JobDescription
from app.services.jd_parser import JDParserService
from app.services.bm25_search import BM25SearchService
from app.services.hybrid_ranker import HybridRankerService
from app.services.embedding_service import EmbeddingService

def run_10k_stress_test():
    print("=" * 75)
    print("      10,000 RESUME ARCHITECTURE & SCALING STRESS TEST")
    print("=" * 75)

    tracemalloc.start()
    t_start_total = time.time()

    jd_parser = JDParserService()
    embedding_service = EmbeddingService()
    bm25_service = BM25SearchService()
    ranker = HybridRankerService()

    sample_jd_text = """Job Title: Senior NLP Machine Learning Engineer
Requirements:
- 3+ years experience with Python, PyTorch, Transformers, NLP, Machine Learning
- Bachelor's or Master's degree in Computer Science
Preferred Skills:
- Docker, FastAPI, Kubernetes, AWS"""
    parsed_jd = jd_parser.parse_job_description(sample_jd_text)

    print("Synthesizing 10,000 structured resume representations...")
    skills_catalogs = [
        ["python", "pytorch", "transformers", "nlp", "machine learning", "deep learning", "docker", "fastapi"],
        ["python", "sql", "pandas", "numpy", "scikit-learn", "xgboost", "aws"],
        ["python", "django", "fastapi", "postgresql", "react", "docker", "git"],
        ["aws", "docker", "kubernetes", "terraform", "ci/cd", "linux"],
        ["financial analysis", "budgeting", "excel", "accounting", "sap"],
        ["recruitment", "talent acquisition", "hr", "communication", "leadership"]
    ]
    degree_options = ["doctorate", "master", "bachelor", "associate", "none"]

    resumes: list[ParsedResume] = []
    t_gen_0 = time.time()
    for i in range(1, 10001):
        cat_idx = (i - 1) % len(skills_catalogs)
        sk = skills_catalogs[cat_idx]
        exp_yr = float((i % 10) + 1)
        deg = degree_options[i % len(degree_options)]
        
        cid = f"CV_{i:05d}"
        clean_text = f"Candidate {cid}. Software professional with {exp_yr:.1f} years experience. Key skills: {', '.join(sk)}. Education tier: {deg}."
        
        r = ParsedResume(
            candidate_id=cid,
            source_file=f"resume_{i:05d}.pdf",
            file_hash=f"hash_{i}",
            text_hash=f"thash_{i}",
            name=f"Candidate {i}",
            raw_text=clean_text,
            cleaned_text=clean_text,
            skills=sk,
            experience_years=exp_yr,
            degrees=[deg],
            section_texts={
                "skills": ", ".join(sk),
                "experience": f"{exp_yr} years in software engineering",
                "projects": "High scale data pipelines",
                "education": deg
            }
        )
        resumes.append(r)
    
    t_gen = time.time() - t_gen_0
    print(f"[1] 10,000 Structured Resumes Prepared: in {t_gen:.2f}s ({10000/t_gen:.0f} resumes/sec)")

    # 2. BM25 INDEXING OVER 10,000 RESUMES
    t_bm25_0 = time.time()
    cids = [r.candidate_id for r in resumes]
    texts = [r.cleaned_text for r in resumes]
    bm25_service.build_index(cids, texts)
    t_bm25 = time.time() - t_bm25_0
    print(f"[2] BM25 Lexical Index Built (10,000 docs): in {t_bm25:.2f}s ({10000/t_bm25:.0f} docs/sec)")

    # 3. VECTOR MAP SYNTHESIS (Normalized 384-d vectors)
    print("Generating vector embedding representations (384-dimensional)...")
    np.random.seed(42)
    # Generate 6 prototype vector embeddings for the 6 career clusters
    proto_vecs = np.random.randn(6, 384).astype(np.float32)
    proto_vecs /= np.linalg.norm(proto_vecs, axis=1, keepdims=True)

    vec_map = {}
    for i, r in enumerate(resumes):
        cat_idx = (i) % 6
        # Slight jitter around cluster centroid
        vec = proto_vecs[cat_idx] + np.random.randn(384).astype(np.float32) * 0.05
        vec /= np.linalg.norm(vec)
        vec_map[r.candidate_id] = {
            "full": vec,
            "skills": vec,
            "experience": vec,
            "projects": vec,
            "education": vec
        }

    # 4. END-TO-END SEARCH OVER 10,000 RESUMES
    print("\nExecuting End-to-End Query over 10,000 Resumes:")
    print("  Pipeline: 10,000 -> Hard Filter -> Hybrid BM25+Semantic -> Top 100 Shortlist")
    
    t_search_0 = time.perf_counter()
    resp = ranker.rank_candidates(
        resumes=resumes,
        resume_vectors_map=vec_map,
        jd=parsed_jd,
        top_k=100
    )
    t_search_ms = (time.perf_counter() - t_search_0) * 1000

    print("\n" + "-" * 75)
    print("10,000-RESUME QUERY METRICS:")
    print(f"Total Resume Pool:              {resp.total_indexed_resumes:,}")
    print(f"Passed Hard Filter:             {resp.hard_filtered_count:,} ({resp.hard_filtered_count/resp.total_indexed_resumes*100:.1f}%)")
    print(f"Candidates Ranked & Scored:     {resp.retrieved_count:,}")
    print(f"Final Shortlist (Top-K):        {resp.final_count}")
    print(f"End-to-End Query Latency:       {t_search_ms:.2f} ms")
    print(f"Query Throughput:               {10000 / (t_search_ms / 1000):.0f} resumes/second")

    print("\nTop 5 Shortlisted Candidates from 10,000 Pool:")
    for c in resp.results[:5]:
        print(f"  Rank #{c.rank:02d} | {c.candidate_id} | Final: {c.final_score:.1f}% | Sem: {c.semantic_score*100:.1f}% | BM25: {c.bm25_score*100:.1f}% | Skills: {c.skill_score*100:.1f}%")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("-" * 75)
    print(f"Peak Memory Overhead for 10k:   {peak_mem / (1024*1024):.2f} MB")
    print(f"Total Test Wallclock Time:      {time.time() - t_start_total:.2f}s")
    print("=" * 75)

if __name__ == "__main__":
    run_10k_stress_test()
